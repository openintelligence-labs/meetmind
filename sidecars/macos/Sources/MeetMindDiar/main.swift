// meetmind-diar-macos — Streaming Sortformer 4spk-v2 sidecar via FluidAudio.
//
// Wire format: same framed protocol as the STT sidecar (SPEC_STT_IPC.md).
//   stdin:  CONTROL frames (JSON-in-frame) + AUDIO_F32 frames
//   stdout: HELLO, CONTROL_ACK, DIAR_SEGMENT (0x13), LOG, BYE

import FluidAudio
import Foundation
import MeetMindIPC

let writer = FrameWriter()

final class StreamDiarizer {
    let stream: StreamId
    let diarizer: SortformerDiarizer
    let frameDurationSeconds: Double
    var lastEmittedSegmentID: Set<UUID> = []

    init(stream: StreamId, models: SortformerModels, config: SortformerConfig) {
        self.stream = stream
        self.diarizer = SortformerDiarizer(config: config)
        self.diarizer.initialize(models: models)
        self.frameDurationSeconds = Double(config.frameDurationSeconds)
    }

    func appendAndProcess(_ samples: [Float]) {
        diarizer.addAudio(samples)
        do {
            if let update = try diarizer.process() {
                emit(update.finalizedSegments)
            }
        } catch {
            writer.log("Diar: process() error on stream \(stream.rawValue): \(error)")
        }
    }

    func flushFinal() {
        do {
            if let update = try diarizer.finalizeSession() {
                emit(update.finalizedSegments)
            }
        } catch {
            writer.log("Diar: finalize error on stream \(stream.rawValue): \(error)")
        }
    }

    private func emit(_ segments: [DiarizerSegment]) {
        for seg in segments where !lastEmittedSegmentID.contains(seg.id) {
            lastEmittedSegmentID.insert(seg.id)
            let startMs = Int(Double(seg.startFrame) * frameDurationSeconds * 1000)
            let endMs = Int(Double(seg.endFrame) * frameDurationSeconds * 1000)
            let streamName = stream == .mic ? "mic" : "loopback"
            writer.writeJSON(
                .diarSegment,
                [
                    "stream":     streamName,
                    "cluster_id": "spk\(seg.speakerIndex)",
                    "start_ms":   startMs,
                    "end_ms":     endMs,
                    "confidence": 0.85,
                ]
            )
        }
    }
}

var sortformerModels: SortformerModels?
var sortformerConfig: SortformerConfig = .default
var diarizers: [StreamId: StreamDiarizer] = [:]

let helloPayload: [String: Any] = [
    "sidecar":          "meetmind-diar-macos",
    "version":          "0.5.0",
    "protocol_version": "1.0.0",
    "platform":         "macos",
    "model":            "diar-streaming-sortformer-4spk-v2",
    "max_speakers":     4,
]
writer.writeJSON(.hello, helloPayload)

func ack(_ id: String, ok: Bool, error: String? = nil) {
    var obj: [String: Any] = ["id": id, "ok": ok]
    if let e = error { obj["error"] = e }
    writer.writeJSON(.controlAck, obj)
}

func loadModelsBlocking() -> Bool {
    let semaphore = DispatchSemaphore(value: 0)
    var ok = true
    Task {
        do {
            writer.log("Diar: downloading + loading Sortformer 4spk-v2…")
            let models = try await SortformerModels.loadFromHuggingFace(config: sortformerConfig)
            sortformerModels = models
            writer.log("Diar: model loaded")
        } catch {
            writer.log("Diar: model load failed: \(error)")
            ok = false
        }
        semaphore.signal()
    }
    semaphore.wait()
    return ok
}

func ensureDiarizer(for stream: StreamId) -> StreamDiarizer? {
    if let d = diarizers[stream] { return d }
    guard let models = sortformerModels else { return nil }
    let d = StreamDiarizer(stream: stream, models: models, config: sortformerConfig)
    diarizers[stream] = d
    return d
}

func handleControl(_ payload: Data) {
    guard
        let cmd = try? JSONSerialization.jsonObject(with: payload) as? [String: Any]
    else { return }
    let id = (cmd["id"] as? String) ?? ""
    let op = (cmd["op"] as? String) ?? ""
    let args = (cmd["args"] as? [String: Any]) ?? [:]

    switch op {
    case "start":
        if sortformerModels == nil {
            let ok = loadModelsBlocking()
            ack(id, ok: ok, error: ok ? nil : "model load failed")
            if !ok { return }
        } else {
            ack(id, ok: true)
        }
    case "flush":
        let streamName = (args["stream"] as? String) ?? "mic"
        let stream: StreamId = streamName == "loopback" ? .loopback : .mic
        diarizers[stream]?.flushFinal()
        ack(id, ok: true)
    case "stop":
        for (_, d) in diarizers { d.flushFinal() }
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
    guard let frame = FrameReader.decodeAudioF32(payload: payload) else { return }
    guard let d = ensureDiarizer(for: frame.stream) else { return }
    d.appendAndProcess(frame.samples)
}

let reader = FrameReader()
while let frame = reader.readFrame() {
    switch frame.type {
    case .control: handleControl(frame.payload)
    case .audio:   handleAudio(frame.payload)
    default:       writer.log("Diar: unexpected frame type \(frame.type.rawValue)")
    }
}

for (_, d) in diarizers { d.flushFinal() }
writer.writeJSON(.bye, ["ok": true])
writer.close()
try? FileHandle.standardOutput.close()
