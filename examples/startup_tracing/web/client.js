import { Room, RoomEvent, Track } from 'livekit-client';
const status = document.querySelector('#status');
let room, context, recorder;
const recorded = [];
const audioNodes = [];
window.result = undefined;
window.traceDebug = () => ({marks:report.marks, state:room?.state, audioState:context?.state,audioTime:context?.currentTime,participants:[...(room?.remoteParticipants?.values()||[])].map(p=>({id:p.identity,tracks:[...p.trackPublications.values()].map(t=>({sid:t.trackSid,subscribed:t.isSubscribed,kind:t.kind,ready:t.track?.mediaStreamTrack?.readyState}))}))});
const clock = () => performance.timeOrigin + performance.now();
const report = { marks: {}, errors: [] };
function mark(name) {
  if (report.marks[name] !== undefined) return;
  report.marks[name] = clock();
  const row = document.createElement('li');
  row.textContent = `${name}: ${(report.marks[name] - report.marks.click).toFixed(1)} ms`;
  document.querySelector('#events').append(row);
}
function paint(name) { requestAnimationFrame(() => requestAnimationFrame(() => mark(name))); }
let probeStarted = false;
function agentState(participant, state) {
  if (!state) return;
  mark('agent_state_'+state);
  status.textContent = `Agent ${state}`;
  paint('agent_state_'+state+'_painted');
  if (['listening','thinking','speaking'].includes(state)) {
    mark('agent_interaction_ready');
    paint('agent_interaction_ready_painted');
    if (!probeStarted && room?.state === 'connected') {
      probeStarted = true;
      mark('first_message_send');
      void room.localParticipant.publishData(new TextEncoder().encode('startup-probe'), {
        reliable: true, topic: 'startup.probe', destinationIdentities: [participant.identity],
      }).then(()=>mark('first_message_sent')).catch(e=>report.errors.push(String(e)));
    }
  }
}

document.querySelector('#start').disabled = true;
async function prepareProbe() {
  context = new AudioContext();
  const processor = `class Detector extends AudioWorkletProcessor {
    process(inputs,outputs) {
    const c=inputs[0]?.[0],o=outputs[0]?.[0];
    if(c&&o)o.set(c);
    if(c&&!this.sent&&c.some(x=>Math.abs(x)>0.005)){this.sent=true;this.port.postMessage({first:true,at:currentTime});}
    return true;
    }
  } registerProcessor('detector',Detector);`;
  const url = URL.createObjectURL(new Blob([processor],{type:'text/javascript'}));
  await context.audioWorklet.addModule(url); URL.revokeObjectURL(url);
  document.querySelector('#start').disabled = false;
  window.probeReady = true;
}
void prepareProbe();
document.querySelector('#start').onclick = async () => {
  document.querySelector('#start').disabled = true;
  mark('click');
  status.textContent = 'Connecting…';
  paint('connecting_ui_painted');
  const runId = new URLSearchParams(location.search).get('run');
  try {
    void context.resume();
    const response = await fetch(`http://127.0.0.1:18790/token/${runId}`);
    const credentials = await response.json(); mark('credentials_ready');
    room = new Room({ singlePeerConnection: true });
    room.on(RoomEvent.Connected,()=>{mark('room_connected');status.textContent='Connected — waiting for agent';paint('connected_ui_painted');});
    room.on(RoomEvent.ParticipantConnected,p=>{if(p.kind===4||p.identity.startsWith('agent-')){mark('agent_visible');paint('agent_visible_ui_painted');agentState(p,p.attributes['lk.agent.state']);}});
    room.on(RoomEvent.ParticipantAttributesChanged,(changed,p)=>{
      if(changed['lk.agent.state'])agentState(p,changed['lk.agent.state']);
    });
    room.on(RoomEvent.DataReceived,(_payload,_participant,_kind,topic)=>{if(topic==='startup.ack')mark('first_message_ack');});
    room.on(RoomEvent.TrackSubscribed,async(track)=>{
      if(track.kind!==Track.Kind.Audio)return;
      mark('track_subscribed');
      const element=track.attach();element.muted=true;document.body.append(element);
      await element.play();mark('audio_element_playing');
      const stream=new MediaStream([track.mediaStreamTrack]);
      recorder=new MediaRecorder(stream);
      recorder.ondataavailable=e=>{if(e.data.size)recorded.push(e.data);};
      recorder.start();
      const source=context.createMediaStreamSource(stream);
      const detector=new AudioWorkletNode(context,'detector');
      source.connect(detector).connect(context.destination);
      audioNodes.push(source,detector);
      detector.port.onmessage=()=>{
        mark('first_rendered_audio');status.textContent='Agent audio is playing';paint('audio_playing_ui_painted');
        setTimeout(async()=>{
          try {
            const pc=room.engine.pcManager.publisher;
            const stats=await pc.getStats();
            const transport=[...stats.values()].find(s=>s.type==='transport'&&s.selectedCandidatePairId);
            const pair=transport&&stats.get(transport.selectedCandidatePairId);
            report.iceRttMs=pair?.currentRoundTripTime*1000;
          }catch(e){report.statsError=String(e);}
          report.audioContextState=context.state;
          report.audioBaseLatencyMs=context.baseLatency*1000;
          await new Promise(resolve=>{recorder.onstop=resolve;recorder.stop();});
          const bytes=new Uint8Array(await new Blob(recorded).arrayBuffer());
          report.audioRecording=btoa(Array.from(bytes,b=>String.fromCharCode(b)).join(''));
          window.result=report;
        },500);
      };
    });
    room.on(RoomEvent.Disconnected,(reason)=>{mark('disconnected');report.disconnectReason=reason;});
    await room.connect(credentials.url,credentials.token);
    mark('connect_returned');
    const agent=[...room.remoteParticipants.values()].find(p=>p.kind===4||p.identity.startsWith('agent-'));
    if(agent){mark('agent_visible');agentState(agent,agent.attributes['lk.agent.state']);}
  }catch(e){report.errors.push(String(e));window.result=report;status.textContent=String(e);}
};
window.cleanup = async()=>{await room?.disconnect();await context?.close();};
