const API_BASE = ''

export function setToken(token) {
  if (token) localStorage.setItem('token', token)
  else localStorage.removeItem('token')
}

export function getToken() {
  return localStorage.getItem('token')
}

async function request(path, { method = 'GET', body, auth = true } = {}) {
  const headers = { 'Content-Type': 'application/json' }
  if (auth) {
    const token = getToken()
    if (token) headers.Authorization = `Bearer ${token}`
  }

  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })

  if (!res.ok) {
    let msg = `Request failed (${res.status})`
    try {
      const data = await res.json()
      msg = data?.detail ?? msg
    } catch {
      // ignore
    }
    throw new Error(msg)
  }
  return await res.json()
}

// Frontend-only Groq key for per-chunk explanations.
// We keep the rest of the interface the same so the doctor UI does not need
// to know which LLM provider is backing the explanation.
const GROQ_API_KEY = import.meta.env.VITE_GROQ_API_KEY
const GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
const GROQ_MODEL = 'llama-3.3-70b-versatile'

async function explainWithGemini(rawText) {
  if (!rawText) {
    return { ok: false, reason: 'empty' }
  }

  // Per-chunk guard: only send up to ~1500 characters to limit cost and
  // reduce the risk of rate limits. The UI always passes a single chunk.
  const text = rawText.slice(0, 1500)

  const prompt = `You are assisting a clinician.

Task:
Explain, in plain language, what the selected guideline excerpt is saying.

Rules:
- Use ONLY the provided text
- Do NOT add recommendations
- Do NOT infer missing information
- Do NOT generalize beyond the text
- If the text is limited or unclear, say so

Tone:
- Clear
- Neutral
- Educational

Output format:
Short paragraph
Bullet points of key ideas

Guideline excerpt:
"""
${text}
"""`

  try {
    if (!GROQ_API_KEY) {
      console.error('Groq API key (VITE_GROQ_API_KEY) is not configured.')
      return { ok: false, reason: 'error' }
    }

    const res = await fetch(GROQ_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${GROQ_API_KEY}`,
      },
      body: JSON.stringify({
        model: GROQ_MODEL,
        messages: [
          { role: 'system', content: 'You are a careful assistant that explains guideline excerpts without adding new clinical recommendations.' },
          { role: 'user', content: prompt },
        ],
        temperature: 0,
      }),
    })

    if (res.status === 429) {
      // Critical: do NOT auto-retry on rate limit. Let the UI show a clear
      // message so the system stays responsive.
      console.warn('Groq API rate limit hit (429). No automatic retry.')
      return { ok: false, reason: 'rate_limit' }
    }

    if (!res.ok) {
      console.error('Groq API request failed:', res.status, res.statusText)
      return { ok: false, reason: 'error' }
    }

    const data = await res.json()
    const explanation = data?.choices?.[0]?.message?.content

    if (!explanation) {
      console.warn('Groq API returned no explanation.')
      return { ok: false, reason: 'empty' }
    }

    // Safety guard: Check for medical advice, recommendations, or new facts.
    // If we detect likely recommendation language, we discard the response
    // so AI can never override guideline text with treatment advice.
    const lowerCaseExplanation = explanation.toLowerCase()
    // Very tight safety guard: only block obviously prescriptive phrasing.
    // We no longer block on generic words like "should" or "consider" to avoid
    // over-filtering reasonable explanations.
    if (
      lowerCaseExplanation.includes('patients should') ||
      lowerCaseExplanation.includes('it is recommended') ||
      lowerCaseExplanation.includes('we recommend') ||
      lowerCaseExplanation.includes('i recommend') ||
      lowerCaseExplanation.includes('you should') ||
      lowerCaseExplanation.includes('treatment should')
    ) {
      console.warn('LLM response flagged for containing explicit recommendation phrasing.')
      return { ok: false, reason: 'safety_block' }
    }

    // Additional safety guard: if the explanation is far longer than the
    // original text, it may be introducing new information.
    if (explanation.length > text.length * 2) {
      console.warn('LLM response flagged for potentially adding new information (length heuristic).')
      return { ok: false, reason: 'safety_block' }
    }

    return { ok: true, text: explanation }
  } catch (e) {
    console.error('Error calling Groq API from frontend:', e)
    return { ok: false, reason: 'error' }
  }
}

export const api = {
  explainWithGemini: (text) => explainWithGemini(text),
  signup: (email, password, role) => request('/api/auth/signup', { method: 'POST', body: { email, password, role }, auth: false }),
  login: (email, password) => request('/api/auth/login', { method: 'POST', body: { email, password }, auth: false }),
  listDocuments: () => request('/api/admin/documents'),
  uploadPdf: async (file) => {
    const token = getToken()
    const form = new FormData()
    form.append('file', file)
    const res = await fetch('/api/admin/upload', {
      method: 'POST',
      headers: token ? { Authorization: `Bearer ${token}` } : {},
      body: form,
    })
    if (!res.ok) {
      let msg = `Upload failed (${res.status})`
      try {
        const data = await res.json()
        msg = data?.detail ?? msg
      } catch {
        // ignore
      }
      throw new Error(msg)
    }
    return await res.json()
  },
  doctorSearch: (query) => request(`/api/doctor/search?query=${encodeURIComponent(query)}`),
}

