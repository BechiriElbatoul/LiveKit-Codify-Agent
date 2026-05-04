import { AccessToken } from 'livekit-server-sdk';
import { NextResponse } from 'next/server';

export async function GET() {
  const token = new AccessToken(
    process.env.LIVEKIT_API_KEY,
    process.env.LIVEKIT_API_SECRET,
    { identity: 'user-' + Date.now() }
  );
  token.addGrant({ roomJoin: true, room: 'my-room', canPublish: true, canSubscribe: true });
  return NextResponse.json({ token: await token.toJwt() });
}
