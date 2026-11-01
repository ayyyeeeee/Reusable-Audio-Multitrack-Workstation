// Check the export controls agree with what the audition gain nodes play.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../stemstudio/web/app.js'), 'utf8');
const definitions = source.slice(0, source.indexOf("$('importButton').addEventListener"));
const context = vm.createContext({});
vm.runInContext(definitions, context);
vm.runInContext(`
state.session = {tracks: [
 {id:'piano',name:'piano',gain:.7,muted:false},
 {id:'removed',name:'removed',gain:1,muted:true},
 {id:'noise',name:'noise',gain:.2,muted:false}
]};`, context);
function snapshot() {
  return JSON.parse(vm.runInContext(`JSON.stringify({
    played:state.session.tracks.map(effectiveGain),
    exported:controls(true).map(t=>t.muted?0:t.gain),
    saved:controls(false).map(t=>t.muted)
  })`, context));
}
let result = snapshot();
assert.deepEqual(result.played, [.7, 0, .2]);
assert.deepEqual(result.played, result.exported);
vm.runInContext("state.solos.add('removed')", context);
result = snapshot();
assert.deepEqual(result.played, [0, 1, 0]);
assert.deepEqual(result.played, result.exported);
assert.deepEqual(result.saved, [false, true, false]);
vm.runInContext("state.solos.add('piano')", context);
result = snapshot();
assert.deepEqual(result.played, [.7, 1, 0]);
assert.deepEqual(result.played, result.exported);
vm.runInContext('state.solos.clear()', context);
assert.deepEqual(snapshot().played, [.7, 0, .2]);
console.log('PASS: mixer/export parity, muted-track audition, multi-solo, saved mute restoration');
