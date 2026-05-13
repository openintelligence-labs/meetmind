// IPC.swift — shared wire-format codec + stdio plumbing.
//
// Mirrors:
//   • meetmind/docs/SPEC_CAPTURE_IPC.md   (capture sidecar protocol)
//   • meetmind/docs/SPEC_STT_IPC.md       (STT sidecar protocol)
//
// Frame layout: [type:u8][length:u32 LE][payload].
// AUDIO payload: [stream:u8][flags:u8][ts:u48 LE][raw PCM].

import Foundation

public enum FrameType: UInt8 {
    case hello       = 0x01
    case ready       = 0x02
    case control     = 0x05  // Python → STT/diar sidecar
    case audio       = 0x10  // capture: PCM s16; STT: PCM f32 (same hex slot)
    case partial     = 0x11  // STT → Python
    case final       = 0x12  // STT → Python
    case diarSegment = 0x13  // diarizer → Python
    case event       = 0x20
    case log         = 0x30
    case controlAck  = 0x40
    case bye         = 0xFF
}

public enum StreamId: UInt8 {
    case mic      = 0x00
    case loopback = 0x01
}

public struct AudioFlags {
    public static let segmentStart: UInt8 = 0b0000_0001
    public static let segmentEnd:   UInt8 = 0b0000_0010
}

/// Thread-safe stdout writer. Survives EPIPE by silently dropping.
public final class FrameWriter {
    private let lock = NSLock()
    private let stdout = FileHandle.standardOutput
    private var closed = false

    public init() {}

    public func close() {
        lock.lock(); defer { lock.unlock() }
        closed = true
    }

    public func write(_ type: FrameType, payload: Data) {
        lock.lock(); defer { lock.unlock() }
        guard !closed else { return }
        var header = Data(capacity: 5)
        header.append(type.rawValue)
        var length = UInt32(payload.count).littleEndian
        withUnsafeBytes(of: &length) { header.append(contentsOf: $0) }
        do {
            try stdout.write(contentsOf: header)
            if !payload.isEmpty {
                try stdout.write(contentsOf: payload)
            }
        } catch {
            closed = true
        }
    }

    public func writeJSON(_ type: FrameType, _ obj: [String: Any]) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []) else {
            return
        }
        write(type, payload: data)
    }

    public func writeAudio(
        stream: StreamId,
        timestampMicros: UInt64,
        pcm: Data,
        segmentStart: Bool = false,
        segmentEnd: Bool = false
    ) {
        var flags: UInt8 = 0
        if segmentStart { flags |= AudioFlags.segmentStart }
        if segmentEnd   { flags |= AudioFlags.segmentEnd }

        var payload = Data(capacity: 8 + pcm.count)
        payload.append(stream.rawValue)
        payload.append(flags)
        var ts = timestampMicros.littleEndian
        let tsBytes = withUnsafeBytes(of: &ts) { Data($0) }
        payload.append(tsBytes.prefix(6))
        payload.append(pcm)
        write(.audio, payload: payload)
    }

    public func log(_ message: String) {
        write(.log, payload: Data(message.utf8))
    }
}

/// Newline-JSON reader used by the capture sidecar (control plane only).
public final class CommandReader {
    public init() {}

    public func readCommand() -> [String: Any]? {
        guard let line = readLine(strippingNewline: true), !line.isEmpty else {
            return nil
        }
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return nil
        }
        return obj
    }
}

/// Decoded frame produced by `FrameReader`.
public struct DecodedFrame {
    public let type: FrameType
    public let payload: Data
}

/// Decoded AUDIO_F32 input — used by the STT sidecar.
public struct AudioF32Frame {
    public let stream: StreamId
    public let timestampMicros: UInt64
    public let samples: [Float]
}

/// Reads length-prefixed binary frames from stdin. Used by the STT
/// sidecar where both CONTROL and AUDIO_F32 frames travel on one pipe.
public final class FrameReader {
    private let stdin = FileHandle.standardInput

    public init() {}

    public func readFrame() -> DecodedFrame? {
        guard let header = readBytes(5) else { return nil }
        let typeByte = header[0]
        let length = UInt32(header[1])
            | (UInt32(header[2]) << 8)
            | (UInt32(header[3]) << 16)
            | (UInt32(header[4]) << 24)
        guard let type = FrameType(rawValue: typeByte) else { return nil }
        let payload: Data
        if length == 0 {
            payload = Data()
        } else {
            guard let data = readBytes(Int(length)) else { return nil }
            payload = data
        }
        return DecodedFrame(type: type, payload: payload)
    }

    public func decodeAudioF32(_ frame: DecodedFrame) -> AudioF32Frame? {
        guard frame.type == .audio || frame.type.rawValue == FrameType.audio.rawValue else {
            return nil
        }
        return Self.decodeAudioF32(payload: frame.payload)
    }

    public static func decodeAudioF32(payload: Data) -> AudioF32Frame? {
        guard payload.count >= 8 else { return nil }
        let streamByte = payload[0]
        guard let stream = StreamId(rawValue: streamByte) else { return nil }
        // u48 LE timestamp at bytes 2..8
        var ts: UInt64 = 0
        for i in 0..<6 {
            ts |= UInt64(payload[2 + i]) << (8 * i)
        }
        let pcm = payload.subdata(in: 8..<payload.count)
        // pcm is little-endian float32, native CPU expected to be LE
        let count = pcm.count / 4
        var samples = [Float](repeating: 0, count: count)
        samples.withUnsafeMutableBufferPointer { buf in
            _ = pcm.copyBytes(to: UnsafeMutableRawBufferPointer(buf))
        }
        return AudioF32Frame(stream: stream, timestampMicros: ts, samples: samples)
    }

    private func readBytes(_ count: Int) -> Data? {
        var collected = Data()
        collected.reserveCapacity(count)
        while collected.count < count {
            guard let chunk = try? stdin.read(upToCount: count - collected.count),
                  !chunk.isEmpty
            else {
                return nil
            }
            collected.append(chunk)
        }
        return collected
    }
}
