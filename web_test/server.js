/**
 * LiveKit Token Server (Node.js / Express)
 * * Description:
 * A lightweight, zero-dependency-bloat server designed to generate LiveKit Access Tokens (JWTs).
 * This implementation prioritizes compatibility and string hygiene to prevent common 
 * encoding issues (BOM, NUL characters) often found when piping data between different environments.
 * * Key Features:
 * - SDK Compatibility: Avoids importing specific `VideoGrant` classes, opting for 
 * primitive object grants to ensure stability across livekit-server-sdk versions.
 * - BOM-Free output: Aggressively sanitizes strings to ensure tokens are clean for client usage.
 * - Explicit Headers: Manages Content-Length manually to prevent encoding ambiguity.
 * * Usage:
 * node server.js
 * * Environment Variables:
 * - LIVEKIT_API_KEY (Required)
 * - LIVEKIT_API_SECRET (Required)
 * - PORT (Optional, default: 3000)
 * - ROOM_NAME (Optional, default: 'playground-quicktest')
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
// Note: We explicitly avoid importing VideoGrant to maintain duck-typing compatibility across SDK versions
const { AccessToken } = require('livekit-server-sdk'); 

const app = express();

// ==============================================================================
// 1. SERVER CONFIGURATION
// ==============================================================================

// Disable Express signatures for minor security/performance benefits
app.set('etag', false);
app.set('x-powered-by', false);

// CORS Configuration: Defaults to wildcard for development ease.
// IN PRODUCTION: Replace origin: "*" with specific client domains.
app.use(cors({
  origin: "*",        
  methods: ["GET"]
}));

const PORT = Number(process.env.PORT || 3000);
const API_KEY = process.env.LIVEKIT_API_KEY;
const API_SECRET = process.env.LIVEKIT_API_SECRET;
const DEFAULT_ROOM = process.env.ROOM_NAME || 'playground-quicktest';

// Fail fast if credentials are missing
if (!API_KEY || !API_SECRET) {
  console.error('Missing LIVEKIT_API_KEY or LIVEKIT_API_SECRET in .env');
  process.exit(1);
}

// ==============================================================================
// 2. UTILITY FUNCTIONS
// ==============================================================================

/**
 * String Sanitizer
 * * Removes invisible characters that often break JWT signing or validation, 
 * specifically the Byte Order Mark (BOM) (\uFEFF) and Null bytes (\u0000).
 * * @param {string|null} str - The raw string input.
 * @returns {string} The cleaned, trimmed string.
 */
const clean = (str) =>
  String(str ?? '')
    .replace(/^\uFEFF/, '') // Remove UTF-8 BOM if present
    .replace(/\u0000/g, '') // Remove Null bytes
    .trim();

/**
 * Generates a LiveKit Access Token.
 * * Wraps the SDK's AccessToken logic to ensure async compatibility and 
 * applies sanitization to user inputs.
 * * @param {Object} params
 * @param {string} [params.room] - Target room name.
 * @param {string} [params.identity] - Unique user identifier.
 * @returns {Promise<string>} The signed JWT string.
 */
async function createToken({ room = DEFAULT_ROOM, identity = null } = {}) {
  const id = clean(identity) || `user-${Math.floor(Math.random() * 1e8)}`;

  // Initialize AccessToken with API credentials
  const at = new AccessToken(API_KEY, API_SECRET, { identity: id });

  // Grant Logic:
  // We pass a plain object to addGrant(). This is the most stable method across
  // major version changes of the SDK, avoiding potential breaking changes in class constructors.
  at.addGrant({
    roomJoin: true,
    room: clean(room),
  });

  // Serialize to JWT (awaiting allows for async crypto implementations in newer Node versions)
  const jwt = await at.toJwt();
  return clean(jwt);
}

// ==============================================================================
// 3. ROUTE HANDLERS
// ==============================================================================

/**
 * GET /browser.token
 * * Generates a token based on query parameters.
 * * Query Params:
 * - room (optional): The room to join.
 * - identity (optional): The username/ID for the session.
 * * Response:
 * - 200: Plain text JWT.
 * - 500: Error message.
 */
app.get('/browser.token', async (req, res) => {
  try {
    const room = clean(req.query.room || DEFAULT_ROOM);
    const identity = clean(req.query.identity || '');

    const token = await createToken({ room, identity });

    // Create a Buffer to accurately calculate Content-Length in bytes.
    // This prevents issues where multi-byte characters might cause a mismatch
    // if we relied on string.length.
    const buffer = Buffer.from(token, 'utf8');
    
    res.setHeader('Content-Type', 'text/plain; charset=utf-8');
    res.setHeader('Content-Length', String(buffer.length));
    return res.end(buffer);
  } catch (err) {
    // Log the full stack trace for server-side debugging
    console.error('Token generation error:', err && err.stack ? err.stack : err);
    
    res
      .status(500)
      .setHeader('Content-Type', 'text/plain; charset=utf-8')
      .end('token_error');
  }
});

// ==============================================================================
// 4. SERVER STARTUP
// ==============================================================================

app.listen(PORT, () => {
  console.log(`Token server running → http://localhost:${PORT}`);
});