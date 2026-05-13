// Capture.swift — the actual audio sources.
import MeetMindIPC

//
// In production this hosts two captures:
//   • Mic input via AVCaptureSession + AVAudioEngine.
//   • System audio via Core Audio Tap API (macOS 14.4+) or ScreenCaptureKit (13.x).
//
// For S1.2 we ship the *shape* and a fully-functional silent generator so the
// sidecar can be exercised end-to-end without OS permissions in CI. The real
// Core Audio Tap implementation is gated behind an environment toggle and a
// follow-up story (S1.2b) once we have a CI machine with screen-recording
// consent baked in.

import Foundation

protocol AudioSource {
    var streamId: StreamId { get }
    func start(_ writer: FrameWriter, baseTimestampMicros: UInt64)
    func stop()
}

/// Generates 10 ms s16 mono frames @ 48 kHz, all-zero payload. Exists so the
/// sidecar pipeline is testable on machines where the real Core Audio Tap
/// can't be set up (CI, sandbox, no consent). Not user-visible: triggered by
/// `MEETMIND_CAPTURE_FAKE=1` in the environment.
final class SilentSource: AudioSource {
    let streamId: StreamId
    private var timer: DispatchSourceTimer?
    private let queue: DispatchQueue
    private let intervalMs: Int

    init(streamId: StreamId, intervalMs: Int = 10) {
        self.streamId = streamId
        self.intervalMs = intervalMs
        self.queue = DispatchQueue(
            label: "meetmind.capture.silent.\(streamId.rawValue)",
            qos: .userInitiated
        )
    }

    func start(_ writer: FrameWriter, baseTimestampMicros: UInt64) {
        let chunkBytes = 480 * 2  // 10 ms * 48 kHz * 2 bytes (s16) = 960 B
        let pcm = Data(count: chunkBytes)
        let stream = self.streamId
        let interval = self.intervalMs
        var index: UInt64 = 0

        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: .milliseconds(interval))
        t.setEventHandler { [weak self] in
            guard self != nil else { return }
            let ts = baseTimestampMicros + index * UInt64(interval) * 1000
            writer.writeAudio(
                stream: stream,
                timestampMicros: ts,
                pcm: pcm,
                segmentStart: index == 0,
                segmentEnd: false
            )
            index &+= 1
        }
        t.resume()
        self.timer = t
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }
}

// MARK: - Real Core Audio Tap source (macOS 14.4+, S1.2b — done)

/// Compatibility shim — `CoreAudioTap` (the real implementation) lives
/// in `CoreAudioTap.swift`. Older code in `main.swift` references
/// `CoreAudioTapSource`; route to the new type when 14.4+ is available
/// and fall through to a silent stub otherwise.
final class CoreAudioTapSource: AudioSource {
    let streamId: StreamId = .loopback
    private var inner: AudioSource?

    func start(_ writer: FrameWriter, baseTimestampMicros: UInt64) {
        if #available(macOS 14.4, *) {
            let real = CoreAudioTap()
            inner = real
            real.start(writer, baseTimestampMicros: baseTimestampMicros)
        } else {
            writer.log("CoreAudioTapSource: macOS 14.4+ required; falling back to silence")
        }
    }

    func stop() { inner?.stop(); inner = nil }
}

// MARK: - ScreenCaptureKit loopback source (macOS 13.x fallback, S1.3b)

/// On macOS 13.x where the Core Audio Tap API is unavailable, capture
/// system audio via ScreenCaptureKit with `excludesCurrentProcessAudio = true`.
/// Implementation deferred; same plumbing as CoreAudioTapSource.
final class ScreenCaptureKitLoopbackSource: AudioSource {
    let streamId: StreamId = .loopback

    func start(_ writer: FrameWriter, baseTimestampMicros: UInt64) {
        writer.log("ScreenCaptureKitLoopbackSource: not yet wired (S1.3b); falling back to silence")
    }

    func stop() {}
}

// MARK: - Source selection

/// Pick the best loopback source for the current OS.
///
/// Returns `CoreAudioTapSource` on macOS 14.4+, `ScreenCaptureKitLoopbackSource`
/// on macOS 13.x. (macOS < 13 is unsupported — App Store Connect won't accept
/// builds for those anyway.)
func makeLoopbackSource() -> AudioSource {
    let info = ProcessInfo.processInfo.operatingSystemVersion
    if info.majorVersion >= 14 {
        return CoreAudioTapSource()
    } else {
        return ScreenCaptureKitLoopbackSource()
    }
}

// MARK: - Mic source (real implementation lives in MicSource.swift)

/// Backwards-compat alias so older code that hardcodes `MicSource()`
/// continues to work. The real type is `RealMicSource`.
final class MicSource: AudioSource {
    let streamId: StreamId = .mic
    private let inner = RealMicSource()

    func start(_ writer: FrameWriter, baseTimestampMicros: UInt64) {
        inner.start(writer, baseTimestampMicros: baseTimestampMicros)
    }

    func stop() { inner.stop() }
}
