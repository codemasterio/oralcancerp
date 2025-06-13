// For production, use relative path (same domain)
// For development, use the full URL
const API_BASE_URL = process.env.NODE_ENV === 'production' 
  ? ''  // Empty string means relative to current domain
  : 'http://localhost:8000';

export {
  API_BASE_URL
};
