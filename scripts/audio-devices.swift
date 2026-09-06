// audio-devices.swift — list and idempotently create the virtual audio devices
// the omnilingual live mode needs: an Aggregate Device "Omnilingual"
// (BlackHole 2ch first, built-in mic last, BlackHole = clock, drift comp on
// the mic) and a stacked Multi-Output Device (built-in speakers + BlackHole).
//
// Commands:
//   list            print "uid | name" for every audio device
//   check           exit 0 iff both virtual devices exist, else 1 (quiet)
//   ensure          create whatever is missing (idempotent), exit nonzero on failure
//   test-lifecycle  create then immediately destroy a throwaway aggregate
//                   (proves the create path works without touching real config)
//   destroy <uid>   destroy one aggregate by UID substring (cleanup)
//
// Uses only stable UID substrings ("BlackHole", "BuiltInMicrophoneDevice",
// "BuiltInSpeakerDevice") with fallbacks to the system default input/output.

import CoreAudio
import Foundation

func getData(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> Data? {
    var addr = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(id, &addr, 0, nil, &size) == noErr, size > 0 else { return nil }
    var data = Data(count: Int(size))
    let status = data.withUnsafeMutableBytes { ptr in
        AudioObjectGetPropertyData(id, &addr, 0, nil, &size, ptr.baseAddress!)
    }
    guard status == noErr else { return nil }
    return data
}

func cfString(_ id: AudioObjectID, _ selector: AudioObjectPropertySelector) -> String? {
    var addr = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<CFString>.size)
    var value: CFString = "" as CFString
    let status = withUnsafeMutablePointer(to: &value) {
        AudioObjectGetPropertyData(id, &addr, 0, nil, &size, UnsafeMutableRawPointer($0))
    }
    guard status == noErr else { return nil }
    return value as String
}

func allDevices() -> [AudioDeviceID] {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDevices,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size) == noErr else { return [] }
    let n = Int(size) / MemoryLayout<AudioDeviceID>.size
    var ids = [AudioDeviceID](repeating: 0, count: n)
    let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &ids)
    guard status == noErr else { return [] }
    return ids
}

func uid(of id: AudioDeviceID) -> String? {
    cfString(id, kAudioDevicePropertyDeviceUID)
}

func name(of id: AudioDeviceID) -> String? {
    cfString(id, kAudioObjectPropertyName)
}

func findDevice(uidContains: String) -> AudioDeviceID? {
    allDevices().first { uid(of: $0)?.contains(uidContains) == true }
}

func findDevice(named: String) -> AudioDeviceID? {
    allDevices().first { name(of: $0) == named }
}

func defaultDevice(isInput: Bool) -> AudioDeviceID? {
    let selector: AudioObjectPropertySelector = isInput
        ? kAudioHardwarePropertyDefaultInputDevice
        : kAudioHardwarePropertyDefaultOutputDevice
    var addr = AudioObjectPropertyAddress(
        mSelector: selector,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var size = UInt32(MemoryLayout<AudioDeviceID>.size)
    var id: AudioDeviceID = 0
    let status = AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &id)
    guard status == noErr, id != 0 else { return nil }
    return id
}

func setRate(_ id: AudioDeviceID, _ rate: Float64, _ label: String) {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyNominalSampleRate,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain)
    var r = rate
    let status = AudioObjectSetPropertyData(id, &addr, 0, nil, UInt32(MemoryLayout<Float64>.size), &r)
    if status != noErr {
        fputs("warning: could not set \(label) sample rate to \(Int(rate)) (in use?); continuing\n", stderr)
    }
}

func createAggregate(name: String, uid: String, mainSubUID: String,
                     subs: [(uid: String, drift: Bool)], stacked: Bool) -> Bool {
    let subDicts: [[String: Any]] = subs.map { s in
        var d: [String: Any] = [kAudioSubDeviceUIDKey as String: s.uid]
        if s.drift { d[kAudioSubDeviceDriftCompensationKey as String] = 1 }
        return d
    }
    let desc: [String: Any] = [
        kAudioAggregateDeviceNameKey as String: name,
        kAudioAggregateDeviceUIDKey as String: uid,
        kAudioAggregateDeviceMainSubDeviceKey as String: mainSubUID,
        kAudioAggregateDeviceIsPrivateKey as String: false,
        kAudioAggregateDeviceIsStackedKey as String: stacked,
        kAudioAggregateDeviceSubDeviceListKey as String: subDicts,
    ]
    var aggID: AudioDeviceID = 0
    let status = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggID)
    if status != noErr {
        fputs("error: AudioHardwareCreateAggregateDevice '\(name)' failed (OSStatus \(status))\n", stderr)
        return false
    }
    return true
}

func destroyAggregate(uid: String) -> Bool {
    guard let id = findDevice(uidContains: uid) else { return false }
    let status = AudioHardwareDestroyAggregateDevice(id)
    if status != noErr {
        fputs("error: AudioHardwareDestroyAggregateDevice '\(uid)' failed (OSStatus \(status))\n", stderr)
        return false
    }
    return true
}

func waitFor(named: String, present: Bool, tries: Int = 50) -> Bool {
    // Aggregate creation/destruction propagates asynchronously; poll briefly.
    for _ in 0..<tries {
        if (findDevice(named: named) != nil) == present { return true }
        usleep(100_000)
    }
    return false
}

func resolveParts() -> (bh: String, mic: String, spk: String)? {
    guard let bh = findDevice(uidContains: "BlackHole").flatMap({ uid(of: $0) }) else {
        fputs("error: BlackHole device not found (install blackhole-2ch, then restart coreaudiod)\n", stderr)
        return nil
    }
    let mic: String? = findDevice(uidContains: "BuiltInMicrophoneDevice").flatMap({ uid(of: $0) })
        ?? defaultDevice(isInput: true).flatMap({ uid(of: $0) })
    let spk: String? = findDevice(uidContains: "BuiltInSpeakerDevice").flatMap({ uid(of: $0) })
        ?? defaultDevice(isInput: false).flatMap({ uid(of: $0) })
    guard let mic, let spk else {
        fputs("error: could not find a built-in mic and speakers\n", stderr)
        return nil
    }
    return (bh, mic, spk)
}

func ensureAll() -> Bool {
    guard let parts = resolveParts() else { return false }
    var ok = true
    if findDevice(named: "Omnilingual") == nil {
        // Order is load-bearing: BlackHole legs first, mic LAST. The live
        // capture downmix selects the last channel as the mic.
        fputs("creating Aggregate Device 'Omnilingual' (BlackHole + mic)…\n", stderr)
        ok = createAggregate(name: "Omnilingual", uid: "omnilingual.aggregate",
                             mainSubUID: parts.bh,
                             subs: [(parts.bh, false), (parts.mic, true)],
                             stacked: false) && ok
        ok = waitFor(named: "Omnilingual", present: true) && ok
    } else {
        fputs("Aggregate Device 'Omnilingual' already exists, skipping\n", stderr)
    }
    if findDevice(named: "Multi-Output Device") == nil {
        fputs("creating 'Multi-Output Device' (speakers + BlackHole)…\n", stderr)
        ok = createAggregate(name: "Multi-Output Device", uid: "omnilingual.multioutput",
                             mainSubUID: parts.spk,
                             subs: [(parts.spk, false), (parts.bh, true)],
                             stacked: true) && ok
        ok = waitFor(named: "Multi-Output Device", present: true) && ok
    } else {
        fputs("'Multi-Output Device' already exists, skipping\n", stderr)
    }
    // Align rates so the aggregate never resamples mid-meeting. Failures are
    // warnings only (a device in use rejects the set).
    if let bhID = findDevice(uidContains: "BlackHole") { setRate(bhID, 48000, "BlackHole") }
    if let micID = findDevice(uidContains: "BuiltInMicrophoneDevice") { setRate(micID, 48000, "mic") }
    if let spkID = findDevice(uidContains: "BuiltInSpeakerDevice") { setRate(spkID, 48000, "speakers") }
    return ok
}

let args = CommandLine.arguments.dropFirst()
let cmd = args.first ?? "ensure"
switch cmd {
case "list":
    for id in allDevices() {
        print("\(uid(of: id) ?? "?") | \(name(of: id) ?? "?")")
    }
case "check":
    exit((findDevice(named: "Omnilingual") != nil && findDevice(named: "Multi-Output Device") != nil) ? 0 : 1)
case "ensure":
    exit(ensureAll() ? 0 : 1)
case "test-lifecycle":
    guard resolveParts() != nil else { exit(1) }
    guard createAggregate(name: "OmnilingualSetupTest", uid: "omnilingual.setuptest",
                          mainSubUID: resolveParts()!.bh,
                          subs: [(resolveParts()!.bh, false)], stacked: false) else { exit(1) }
    guard waitFor(named: "OmnilingualSetupTest", present: true) else {
        fputs("error: test aggregate created but not visible\n", stderr)
        exit(1)
    }
    print("created OmnilingualSetupTest")
    guard destroyAggregate(uid: "omnilingual.setuptest") else { exit(1) }
    guard waitFor(named: "OmnilingualSetupTest", present: false) else {
        fputs("error: test aggregate destroyed but still visible\n", stderr)
        exit(1)
    }
    print("destroyed OmnilingualSetupTest")
case "destroy":
    guard let target = args.dropFirst().first else {
        fputs("usage: audio-devices destroy <uid-substring>\n", stderr)
        exit(2)
    }
    exit(destroyAggregate(uid: target) ? 0 : 1)
default:
    fputs("usage: audio-devices [list|check|ensure|test-lifecycle|destroy <uid>]\n", stderr)
    exit(2)
}
