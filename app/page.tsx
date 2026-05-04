'use client';
import { useState, useCallback } from 'react';
import { LiveKitRoom, RoomAudioRenderer, useVoiceAssistant } from '@livekit/components-react';

export default function Page() {
  const [token, setToken] = useState<string | null>(null);

  const connect = useCallback(async () => {
    const res = await fetch('/api/token');
    const { token } = await res.json();
    setToken(token);
  }, []);

  return (
    <main style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:'100vh', gap:'2rem', fontFamily:'sans-serif' }}>
      <div style={{ textAlign:'center' }}>
        <p style={{ fontSize:12, letterSpacing:'0.08em', color:'#888', textTransform:'uppercase' }}>City General Hospital</p>
        <p style={{ fontSize:22, fontWeight:500, marginTop:4 }}>Sarah</p>
        <p style={{ fontSize:14, color:'#888', marginTop:2 }}>Patient intake agent</p>
      </div>

      {!token ? (
        <CallButton onClick={connect} active={false} />
      ) : (
        <LiveKitRoom serverUrl={process.env.NEXT_PUBLIC_LIVEKIT_URL} token={token} connect audio video={false}>
          <ActiveCall onHangup={() => setToken(null)} />
          <RoomAudioRenderer />
        </LiveKitRoom>
      )}
    </main>
  );
}

function ActiveCall({ onHangup }: { onHangup: () => void }) {
  const { state } = useVoiceAssistant();
  return <CallButton onClick={onHangup} active={true} label={state === 'speaking' ? 'Sarah is speaking…' : 'Listening…'} />;
}

function CallButton({ onClick, active, label }: { onClick: () => void; active: boolean; label?: string }) {
  return (
    <>
      <button onClick={onClick} style={{ width:80, height:80, borderRadius:'50%', background: active ? '#E24B4A' : '#1D9E75', border:'none', cursor:'pointer', display:'flex', alignItems:'center', justifyContent:'center' }}>
        {active
          ? <svg width="32" height="32" viewBox="0 0 24 24" fill="white"><path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z"/></svg>
          : <svg width="32" height="32" viewBox="0 0 24 24" fill="white"><path d="M6.6 10.8c1.4 2.8 3.8 5.1 6.6 6.6l2.2-2.2c.3-.3.7-.4 1-.2 1.1.4 2.3.6 3.6.6.6 0 1 .4 1 1V20c0 .6-.4 1-1 1C10.6 21 3 13.4 3 4c0-.6.4-1 1-1h3.5c.6 0 1 .4 1 1 0 1.3.2 2.5.6 3.6.1.3 0 .7-.2 1L6.6 10.8z"/></svg>
        }
      </button>
      <p style={{ fontSize:13, color:'#888' }}>{label ?? 'Tap to start'}</p>
    </>
  );
}