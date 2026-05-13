// MicSource.swift — real microphone capture via AVAudioEngine.
//
// Default input device → 48kHz mono s16 LE PCM frames → FrameWriter.
// Frames are emitted in 10ms chunks to match the wire-format cadence.
//
// Threading: AVAudioEngine delivers buffers on its render thread; we
// resample/convert and forward via FrameWriter (which is locked).

import AVFoundation
import Foundation
import MeetMindIPC

final class RealMicSource: AudioSource {
    let streamId: StreamId = .mic

    private let engine = AVAudioEngine()
    private var converter: AVAudioConverter?
    private var targetFormat: AVAudioFormat?
    private var startTimeMicros: UInt64 = 0
    private var frameCount: Int = 0
    private var weakWriter: FrameWriter?

    func start(_ writer: FrameWriter, baseTimestampMicros: UInt64) {
        weakWriter = writer
        startTimeMicros = baseTimestampMicros

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)

        guard
            let outFormat = AVAudioFormat(
                commonFormat: .pcmFormatInt16,
                sampleRate: 48_000,
                channels: 1,
                interleaved: true
            )
        else {
            writer.log("MicSource: failed to construct 48kHz s16 mono format")
            return
        }
        targetFormat = outFormat
        converter = AVAudioConverter(from: inputFormat, to: outFormat)
        if converter == nil {
            writer.log("MicSource: AVAudioConverter init failed (input=\(inputFormat))")
            return
        }

        let bufferSize: AVAudioFrameCount = 480  // 10 ms @ 48 kHz
        input.installTap(
            onBus: 0,
            bufferSize: bufferSize,
            format: inputFormat
        ) { [weak self] buffer, _ in
            self?.handleInputBuffer(buffer)
        }

        do {
            try engine.start()
            writer.log("MicSource: AVAudioEngine started (input=\(inputFormat.sampleRate) Hz)")
        } catch {
            writer.log("MicSource: failed to start engine: \(error)")
            engine.inputNode.removeTap(onBus: 0)
        }
    }

    private func handleInputBuffer(_ buffer: AVAudioPCMBuffer) {
        guard
            let writer = weakWriter,
            let target = targetFormat,
            let converter = converter
        else {
            return
        }

        // Convert to 48 kHz mono s16. Output capacity in frames at target rate.
        let inputFrames = Double(buffer.frameLength)
        let inputRate = buffer.format.sampleRate
        let outputCapacity = AVAudioFrameCount(
            ceil(inputFrames * (target.sampleRate / inputRate)) + 16
        )
        guard
            let outBuffer = AVAudioPCMBuffer(
                pcmFormat: target,
                frameCapacity: outputCapacity
            )
        else {
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
            return buffer
        }
        if status == .error {
            return
        }

        let frames = Int(outBuffer.frameLength)
        guard frames > 0, let int16Channel = outBuffer.int16ChannelData else { return }
        let pcmCount = frames * MemoryLayout<Int16>.size
        let pcm = Data(bytes: int16Channel[0], count: pcmCount)

        // Wall-clock timestamp (μs from session start).
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
        if engine.isRunning {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        weakWriter = nil
    }
}
