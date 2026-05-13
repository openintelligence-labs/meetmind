// meetmind-stt-macos — Parakeet TDT 0.6B v3 sidecar via FluidAudio.
//
// Wire format: see meetmind/docs/SPEC_STT_IPC.md.
//   stdin:  CONTROL frames (JSON-in-frame) + AUDIO_F32 frames (raw float32 LE)
//   stdout: HELLO, CONTROL_ACK, PARTIAL, FINAL, LOG, BYE

import AVFoundation
import FluidAudio
import Foundation
import MeetMindIPC

// 16 kHz mono is the FluidAudio target; matches our SPEC_STT_IPC.md AUDIO_F32 contract.
let kSampleRate: Double = 16_000

// 4s of audio per transcription chunk — TDT decodes at RTFx > 1k on Apple
// Silicon, so 4s of buffered audio finishes well under 100ms wall-clock.
// Smaller chunks give snappier FINAL turnaround on short utterances; the
// model's accuracy is unaffected at this size.
let kChunkSamples = 4 * Int(kSampleRate)

// Per-stream PCM buffer + decoder state.
final class StreamBuffer {
    var samples: [Float] = []
    var startMs: Int = 0
    var totalMsConsumed: Int = 0
    var decoderState: TdtDecoderState?
}

let writer = FrameWriter()
var asr: AsrManager?
var modelLoaded = false
var buffers: [StreamId: StreamBuffer] = [.mic: StreamBuffer(), .loopback: StreamBuffer()]

let helloPayload: [String: Any] = [
    "sidecar":          "meetmind-stt-macos",
    "version":          "0.5.0",
    "protocol_version": "1.0.0",
    "platform":         "macos",
    "model":            "parakeet-tdt-0.6b-v3",
    "sample_rate":      Int(kSampleRate),
    "languages":        ["en"],
]
writer.writeJSON(.hello, helloPayload)

// MARK: - control helpers

func ack(_ id: String, ok: Bool, error: String? = nil, data: [String: Any]? = nil) {
    var obj: [String: Any] = ["id": id, "ok": ok]
    if let e = error { obj["error"] = e }
    if let d = data { obj["data"] = d }
    writer.writeJSON(.controlAck, obj)
}

func emitFinal(stream: StreamId, text: String, startMs: Int, endMs: Int, confidence: Double) {
    let streamName = stream == .mic ? "mic" : "loopback"
    writer.writeJSON(
        .final,
        [
            "stream":         streamName,
            "text":           text,
            "start_ms":       startMs,
            "end_ms":         endMs,
            "confidence":     confidence,
            "language":       "en",
        ]
    )
}

// MARK: - transcription

func loadModelsBlocking() {
    let semaphore = DispatchSemaphore(value: 0)
    var loadError: Error?
    Task {
        do {
            writer.log("STT: downloading + loading Parakeet TDT 0.6B v3 (first run may take a while)…")
            let models = try await AsrModels.downloadAndLoad()
            let manager = AsrManager()
            try await manager.loadModels(models)
            asr = manager
            modelLoaded = true
            writer.log("STT: model loaded")
        } catch {
            loadError = error
            writer.log("STT: model load failed: \(error)")
        }
        semaphore.signal()
    }
    semaphore.wait()
    if loadError != nil { exit(3) }
}

func transcribeBuffer(_ stream: StreamId, force: Bool) {
    guard let buf = buffers[stream], let manager = asr else { return }
    if buf.samples.isEmpty { return }
    if !force && buf.samples.count < kChunkSamples { return }

    let chunkSamples = buf.samples
    let chunkStartMs = buf.startMs
    let chunkDurationMs = Int((Double(chunkSamples.count) / kSampleRate) * 1000)
    let chunkEndMs = chunkStartMs + chunkDurationMs

    buf.samples.removeAll(keepingCapacity: true)
    buf.startMs = chunkEndMs
    buf.totalMsConsumed += chunkDurationMs

    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            if buf.decoderState == nil {
                buf.decoderState = try TdtDecoderState()
            }
            var decoderState = buf.decoderState!
            let result = try await manager.transcribe(chunkSamples, decoderState: &decoderState)
            buf.decoderState = decoderState
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                emitFinal(
                    stream: stream,
                    text: text,
                    startMs: chunkStartMs,
                    endMs: chunkEndMs,
                    confidence: Double(result.confidence)
                )
            }
        } catch {
            writer.log("STT: transcribe error on stream \(stream.rawValue): \(error)")
        }
        semaphore.signal()
    }
    semaphore.wait()
}

func handleControl(_ payload: Data) {
    guard
        let cmd = try? JSONSerialization.jsonObject(with: payload) as? [String: Any]
    else {
        return
    }
    let id = (cmd["id"] as? String) ?? ""
    let op = (cmd["op"] as? String) ?? ""
    let args = (cmd["args"] as? [String: Any]) ?? [:]

    switch op {
    case "start":
        if !modelLoaded {
            loadModelsBlocking()
        }
        ack(id, ok: modelLoaded, error: modelLoaded ? nil : "model load failed")
    case "flush":
        let streamName = (args["stream"] as? String) ?? "mic"
        let stream: StreamId = streamName == "loopback" ? .loopback : .mic
        transcribeBuffer(stream, force: true)
        ack(id, ok: true)
    case "stop":
        for stream in buffers.keys {
            transcribeBuffer(stream, force: true)
        }
        ack(id, ok: true)
        writer.writeJSON(.bye, ["ok": true])
        writer.close()
        try? FileHandle.standardOutput.close()
        exit(0)
    default:
        ack(id, ok: false, error: "unknown op: \(op)")
    }
}

func handleAudio(_ payload: Data) {
    guard let frame = FrameReader.decodeAudioF32(payload: payload) else {
        return
    }
    guard let buf = buffers[frame.stream] else { return }
    if buf.samples.isEmpty {
        buf.startMs = buf.totalMsConsumed
    }
    buf.samples.append(contentsOf: frame.samples)
    transcribeBuffer(frame.stream, force: false)
}

// MARK: - input loop

let reader = FrameReader()

while let frame = reader.readFrame() {
    switch frame.type {
    case .control:
        handleControl(frame.payload)
    case .audio:
        handleAudio(frame.payload)
    default:
        writer.log("STT: unexpected frame type \(frame.type.rawValue) on stdin")
    }
}

for stream in buffers.keys {
    transcribeBuffer(stream, force: true)
}
writer.writeJSON(.bye, ["ok": true])
writer.close()
try? FileHandle.standardOutput.close()
