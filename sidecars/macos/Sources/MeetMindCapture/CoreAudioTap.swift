// CoreAudioTap.swift — system-audio capture via Core Audio Tap (macOS 14.4+).
//
// Reference: Apple's "Capturing system audio with Core Audio taps" guide and
// the audiotee + AudioCap open-source samples.
//
// Lifecycle:
//   1. Build a CATapDescription that excludes our own PID and taps the
//      default output device.
//   2. Create the process tap with `AudioHardwareCreateProcessTap`.
//   3. Wrap the tap in an aggregate device so we get a stable stream
//      format and an IOProc to pull frames.
//   4. Start an `AudioDeviceIOProcID` that drives a callback delivering
//      48 kHz s16 (or float32) frames, which we convert + forward.
//
// Permissions: requires Screen Recording (macOS 15+ extends the TCC
// gate to system audio taps). If not granted, we emit a single LOG
// frame and exit early — the sidecar exits 2 in main.swift if the
// permission probe came back denied.

import AVFoundation
import CoreAudio
import Foundation
import MeetMindIPC

@available(macOS 14.4, *)
final class CoreAudioTap: AudioSource {
    let streamId: StreamId = .loopback

    private var tapID: AudioObjectID = 0
    private var aggregateDeviceID: AudioObjectID = 0
    private var ioProcID: AudioDeviceIOProcID?
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?
    private var sourceFormat: AVAudioFormat?
    private var startTimeMicros: UInt64 = 0
    private var frameCount: Int = 0
    private weak var weakWriter: FrameWriter?

    func start(_ writer: FrameWriter, baseTimestampMicros: UInt64) {
        weakWriter = writer
        startTimeMicros = baseTimestampMicros

        guard Permissions.screenRecording() == .granted else {
            writer.log(
                "CoreAudioTap: Screen Recording permission not granted; "
                + "system audio capture skipped."
            )
            return
        }

        // 1. Tap description — capture the global system mix, excluding
        // our own process. NB: the `processes` array takes
        // **AudioObjectID**s of process objects (not raw POSIX PIDs).
        // We translate via kAudioHardwarePropertyTranslatePIDToProcessObject.
        let selfPid = ProcessInfo.processInfo.processIdentifier
        let excludeObjects: [AudioObjectID]
        if let selfObject = translatePID(toProcessObject: selfPid) {
            excludeObjects = [selfObject]
        } else {
            // No self-exclusion possible — capture everything (we'll get
            // a small echo of our own log lines, but they're empty PCM).
            excludeObjects = []
        }
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: excludeObjects)
        description.name = "MeetMindCaptureTap"
        description.isPrivate = true

        var newTapID: AudioObjectID = 0
        let createStatus = AudioHardwareCreateProcessTap(description, &newTapID)
        guard createStatus == noErr else {
            writer.log("CoreAudioTap: AudioHardwareCreateProcessTap failed: \(createStatus)")
            return
        }
        tapID = newTapID

        // The aggregate device's tap-list refers to taps by their actual
        // UID (a CFString assigned by Core Audio when the tap is
        // created). Earlier versions of this code passed a made-up
        // string here, which silently produced an aggregate device
        // wrapping a non-existent tap — IOProc never fired. Query the
        // real UID via kAudioTapPropertyUID.
        guard let tapUID = stringProperty(of: newTapID, selector: kAudioTapPropertyUID) else {
            writer.log("CoreAudioTap: cannot read tap UID")
            destroyTap()
            return
        }

        // 2. Discover default output UID (we wrap the tap in an aggregate
        // device that mirrors the default output's format).
        guard let outputUID = defaultOutputDeviceUID() else {
            writer.log("CoreAudioTap: cannot resolve default output device UID")
            destroyTap()
            return
        }

        // 3. Build an aggregate device wrapping just our tap. Apple's
        // audiotee sample uses this exact shape (name + UID + clock +
        // sub-device list + tap list, no `MainSubDevice`).
        let uniqueAggUID = "com.openintelligence.meetmind.capture.\(UUID().uuidString)"
        let aggregateDescription: [String: Any] = [
            kAudioAggregateDeviceNameKey as String: "MeetMind Capture Aggregate",
            kAudioAggregateDeviceUIDKey as String: uniqueAggUID,
            kAudioAggregateDeviceClockDeviceKey as String: outputUID,
            kAudioAggregateDeviceIsPrivateKey as String: true,
            kAudioAggregateDeviceTapListKey as String: [
                [
                    kAudioSubTapUIDKey as String: tapUID,
                    kAudioSubTapDriftCompensationKey as String: true,
                ]
            ],
        ]

        var aggID: AudioObjectID = 0
        let aggStatus = AudioHardwareCreateAggregateDevice(
            aggregateDescription as CFDictionary,
            &aggID
        )
        guard aggStatus == noErr else {
            writer.log("CoreAudioTap: AudioHardwareCreateAggregateDevice failed: \(aggStatus)")
            destroyTap()
            return
        }
        aggregateDeviceID = aggID

        // 4. Resolve the source stream format. Prefer the tap's own
        // format property — for capture-only aggregate devices the
        // aggregate's stream-format property may not be populated.
        let asbdSource =
            tapStreamFormat(tapID: newTapID)
            ?? streamFormat(of: aggID, scope: kAudioObjectPropertyScopeInput)
            ?? streamFormat(of: aggID, scope: kAudioObjectPropertyScopeOutput)
        guard var asbd = asbdSource else {
            writer.log("CoreAudioTap: cannot read aggregate/tap stream format")
            tearDownAggregate()
            destroyTap()
            return
        }
        sourceFormat = AVAudioFormat(streamDescription: &asbd.copy)
        guard
            let target = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: 48_000,
                channels: 1,
                interleaved: true
            ),
            let src = sourceFormat,
            let conv = AVAudioConverter(from: src, to: target)
        else {
            writer.log("CoreAudioTap: failed to build AVAudioConverter")
            tearDownAggregate()
            destroyTap()
            return
        }
        targetFormat = target
        converter = conv

        // 5. Install IOProc.
        var procID: AudioDeviceIOProcID?
        let ioStatus = AudioDeviceCreateIOProcIDWithBlock(
            &procID,
            aggID,
            nil
        ) { [weak self] _, inInputData, _, _, _ in
            self?.handleAudio(inInputData)
        }
        guard ioStatus == noErr, let procID = procID else {
            writer.log("CoreAudioTap: AudioDeviceCreateIOProcIDWithBlock failed: \(ioStatus)")
            tearDownAggregate()
            destroyTap()
            return
        }
        ioProcID = procID

        let startStatus = AudioDeviceStart(aggID, procID)
        if startStatus != noErr {
            writer.log("CoreAudioTap: AudioDeviceStart failed: \(startStatus)")
            tearDown()
            return
        }
        writer.log(
            "CoreAudioTap: started (source=\(asbd.copy.mSampleRate) Hz, "
            + "\(asbd.copy.mChannelsPerFrame) ch)"
        )
    }

    private func handleAudio(_ data: UnsafePointer<AudioBufferList>) {
        guard
            let writer = weakWriter,
            let target = targetFormat,
            let source = sourceFormat,
            let converter = converter
        else { return }

        let bufferList = UnsafeMutableAudioBufferListPointer(
            UnsafeMutablePointer(mutating: data)
        )
        guard let firstBuffer = bufferList.first, firstBuffer.mDataByteSize > 0 else { return }

        let bytesPerFrame = source.streamDescription.pointee.mBytesPerFrame
        guard bytesPerFrame > 0 else { return }
        let inputFrameCount = Int(firstBuffer.mDataByteSize) / Int(bytesPerFrame)
        guard
            let inBuffer = AVAudioPCMBuffer(
                pcmFormat: source,
                frameCapacity: AVAudioFrameCount(inputFrameCount)
            )
        else { return }
        inBuffer.frameLength = AVAudioFrameCount(inputFrameCount)

        // Copy raw bytes from the IO buffer list into the AVAudioPCMBuffer.
        if let dst = inBuffer.audioBufferList.pointee.mBuffers.mData,
            let src = firstBuffer.mData {
            memcpy(dst, src, Int(firstBuffer.mDataByteSize))
        }

        let outCapacity = AVAudioFrameCount(
            ceil(Double(inputFrameCount) * (target.sampleRate / source.sampleRate)) + 16
        )
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outCapacity) else {
            return
        }

        var inputProvided = false
        let status = converter.convert(to: outBuffer, error: nil) { _, outStatus in
            if inputProvided {
                outStatus.pointee = .noDataNow
                return nil
            }
            inputProvided = true
            outStatus.pointee = .haveData
            return inBuffer
        }
        if status == .error || outBuffer.frameLength == 0 {
            return
        }

        let frames = Int(outBuffer.frameLength)
        guard let int16Channel = outBuffer.int16ChannelData else { return }
        let pcm = Data(bytes: int16Channel[0], count: frames * MemoryLayout<Int16>.size)

        let now = UInt64(Date().timeIntervalSince1970 * 1_000_000)
        let ts = now - startTimeMicros
        let isFirst = (frameCount == 0)
        frameCount += 1

        writer.writeAudio(
            stream: streamId,
            timestampMicros: ts,
            pcm: pcm,
            segmentStart: isFirst,
            segmentEnd: false
        )
    }

    func stop() {
        tearDown()
        weakWriter = nil
    }

    private func tearDown() {
        if let procID = ioProcID, aggregateDeviceID != 0 {
            AudioDeviceStop(aggregateDeviceID, procID)
            AudioDeviceDestroyIOProcID(aggregateDeviceID, procID)
        }
        ioProcID = nil
        tearDownAggregate()
        destroyTap()
    }

    private func tearDownAggregate() {
        if aggregateDeviceID != 0 {
            AudioHardwareDestroyAggregateDevice(aggregateDeviceID)
            aggregateDeviceID = 0
        }
    }

    private func destroyTap() {
        if tapID != 0 {
            AudioHardwareDestroyProcessTap(tapID)
            tapID = 0
        }
    }

    // MARK: - helpers

    private func defaultOutputDeviceUID() -> String? {
        var defaultDevice: AudioObjectID = 0
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultOutputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &addr,
            0,
            nil,
            &size,
            &defaultDevice
        )
        guard status == noErr else { return nil }

        var uidCFString: CFString? = nil
        var uidSize = UInt32(MemoryLayout<CFString?>.size)
        var uidAddr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let uidStatus = AudioObjectGetPropertyData(
            defaultDevice,
            &uidAddr,
            0,
            nil,
            &uidSize,
            &uidCFString
        )
        guard uidStatus == noErr, let cf = uidCFString else { return nil }
        return cf as String
    }

    private func translatePID(toProcessObject pid: pid_t) -> AudioObjectID? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var inputPid = pid
        var processObject: AudioObjectID = 0
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject),
            &addr,
            UInt32(MemoryLayout<pid_t>.size),
            &inputPid,
            &size,
            &processObject
        )
        guard status == noErr, processObject != 0 else { return nil }
        return processObject
    }

    private func stringProperty(
        of object: AudioObjectID,
        selector: AudioObjectPropertySelector
    ) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var cfString: Unmanaged<CFString>? = nil
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let status = AudioObjectGetPropertyData(object, &addr, 0, nil, &size, &cfString)
        guard status == noErr, let value = cfString?.takeRetainedValue() else { return nil }
        return value as String
    }

    private func streamFormat(
        of device: AudioObjectID,
        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeOutput
    ) -> ASBDBox? {
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var asbd = AudioStreamBasicDescription()
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamFormat,
            mScope: scope,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(
            device,
            &addr,
            0,
            nil,
            &size,
            &asbd
        )
        guard status == noErr, asbd.mSampleRate > 0, asbd.mChannelsPerFrame > 0 else {
            return nil
        }
        return ASBDBox(copy: asbd)
    }

    private func tapStreamFormat(tapID: AudioObjectID) -> ASBDBox? {
        var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
        var asbd = AudioStreamBasicDescription()
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioTapPropertyFormat,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd)
        guard status == noErr, asbd.mSampleRate > 0, asbd.mChannelsPerFrame > 0 else {
            return nil
        }
        return ASBDBox(copy: asbd)
    }
}

/// Tiny container so we can hand the ASBD around as a value type with a
/// `&copy` pointer for the AVAudioFormat init.
struct ASBDBox {
    var copy: AudioStreamBasicDescription
}
