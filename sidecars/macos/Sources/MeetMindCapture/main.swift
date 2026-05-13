// main.swift — sidecar entrypoint.
//
// Lifecycle:
//   1. Emit HELLO frame with capabilities + permissions.
//   2. Read newline-JSON commands from stdin.
//   3. On `start`: emit READY frames, begin streaming AUDIO.
//   4. On `stop`: emit BYE, exit 0.
//
// Permission model: if a required permission is `.denied` or `.restricted`
// at start time, exit 2 (Python surfaces a UI prompt).

import Foundation
import MeetMindIPC

let writer = FrameWriter()
let reader = CommandReader()

let useFake = ProcessInfo.processInfo.environment["MEETMIND_CAPTURE_FAKE"] == "1"

// ---- HELLO ----
let permsSnapshot: [String: String] = [
    "microphone":   Permissions.microphone().rawValue,
    "screen_audio": Permissions.screenRecording().rawValue,
]

let hello: [String: Any] = [
    "sidecar":          "meetmind-capture-macos",
    "version":          "0.4.0",
    "protocol_version": "1.0.0",
    "platform":         "macos",
    "capabilities":     ["mic", "loopback"],
    "permissions":      permsSnapshot,
]
writer.writeJSON(.hello, hello)

// Sources.
var sources: [StreamId: AudioSource] = [:]
let baseTs = UInt64(Date().timeIntervalSince1970 * 1_000_000)

func ack(_ id: String, ok: Bool, error: String? = nil, data: [String: Any]? = nil) {
    var obj: [String: Any] = ["id": id, "ok": ok]
    if let e = error { obj["error"] = e }
    if let d = data  { obj["data"] = d }
    writer.writeJSON(FrameType.controlAck, obj)
}

func makeSource(forStream stream: StreamId) -> AudioSource {
    if useFake {
        return SilentSource(streamId: stream)
    }
    switch stream {
    case .mic:      return MicSource()
    case .loopback: return makeLoopbackSource()  // 14+ → tap; 13.x → SCK
    }
}

// ---- Command loop ----
while let cmd = reader.readCommand() {
    let id = (cmd["id"] as? String) ?? ""
    let op = (cmd["op"] as? String) ?? ""
    let args = (cmd["args"] as? [String: Any]) ?? [:]

    switch op {
    case "start":
        let requested = (args["streams"] as? [String]) ?? ["mic", "loopback"]
        var ready: [(StreamId, String)] = []
        for s in requested {
            guard let stream = (s == "mic" ? StreamId.mic : (s == "loopback" ? StreamId.loopback : nil)) else {
                continue
            }
            let src = makeSource(forStream: stream)
            sources[stream] = src
            ready.append((stream, s))
        }
        ack(id, ok: true)
        for (stream, name) in ready {
            writer.writeJSON(.ready, [
                "stream_id": name,
                "format": ["rate": 48000, "encoding": "s16le", "channels": 1],
            ])
            sources[stream]?.start(writer, baseTimestampMicros: baseTs)
        }
    case "stop":
        // Stop sources FIRST so timer threads stop racing with our writes.
        for (_, src) in sources { src.stop() }
        sources.removeAll()
        // Brief settle window: any inflight timer fires drop on `closed`.
        Thread.sleep(forTimeInterval: 0.02)
        ack(id, ok: true)
        writer.writeJSON(.bye, [
            "ok": true,
            "duration_ms": Int((Date().timeIntervalSince1970 * 1000.0) - (Double(baseTs) / 1000.0)),
        ])
        writer.close()
        try? FileHandle.standardOutput.close()
        exit(0)
    case "pause":
        for (_, src) in sources { src.stop() }
        ack(id, ok: true)
    case "resume":
        for (_, src) in sources { src.start(writer, baseTimestampMicros: baseTs) }
        ack(id, ok: true)
    case "query":
        let what = (args["what"] as? String) ?? ""
        switch what {
        case "permissions":
            ack(id, ok: true, data: ["permissions": permsSnapshot])
        default:
            ack(id, ok: false, error: "unknown query: \(what)")
        }
    default:
        ack(id, ok: false, error: "unknown op: \(op)")
    }
}

// EOF on stdin → graceful shutdown.
for (_, src) in sources { src.stop() }
sources.removeAll()
Thread.sleep(forTimeInterval: 0.02)
writer.writeJSON(.bye, ["ok": true])
writer.close()
try? FileHandle.standardOutput.close()
exit(0)
