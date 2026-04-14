export const SECTOR_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  'Technology':             { bg: 'rgba(59,130,246,0.12)',  text: '#60A5FA',  border: 'rgba(59,130,246,0.25)' },
  'Industrials':            { bg: 'rgba(148,163,184,0.12)', text: '#94A3B8',  border: 'rgba(148,163,184,0.25)' },
  'Financial Services':     { bg: 'rgba(234,179,8,0.12)',   text: '#FACC15',  border: 'rgba(234,179,8,0.25)' },
  'Consumer Cyclical':      { bg: 'rgba(249,115,22,0.12)',  text: '#FB923C',  border: 'rgba(249,115,22,0.25)' },
  'Healthcare':             { bg: 'rgba(168,85,247,0.12)',  text: '#C084FC',  border: 'rgba(168,85,247,0.25)' },
  'Energy':                 { bg: 'rgba(239,68,68,0.12)',   text: '#F87171',  border: 'rgba(239,68,68,0.25)' },
  'Consumer Defensive':     { bg: 'rgba(34,197,94,0.12)',   text: '#4ADE80',  border: 'rgba(34,197,94,0.25)' },
  'Real Estate':            { bg: 'rgba(244,114,182,0.12)', text: '#F472B6',  border: 'rgba(244,114,182,0.25)' },
  'Communication Services': { bg: 'rgba(56,189,248,0.12)',  text: '#38BDF8',  border: 'rgba(56,189,248,0.25)' },
  'Utilities':              { bg: 'rgba(163,230,53,0.12)',  text: '#A3E635',  border: 'rgba(163,230,53,0.25)' },
  'Basic Materials':        { bg: 'rgba(45,212,191,0.12)',  text: '#2DD4BF',  border: 'rgba(45,212,191,0.25)' },
  'Financials':             { bg: 'rgba(234,179,8,0.12)',   text: '#FACC15',  border: 'rgba(234,179,8,0.25)' },
  'Consumer Discretionary': { bg: 'rgba(249,115,22,0.12)',  text: '#FB923C',  border: 'rgba(249,115,22,0.25)' },
  'Consumer Staples':       { bg: 'rgba(34,197,94,0.12)',   text: '#4ADE80',  border: 'rgba(34,197,94,0.25)' },
  'Health Care':            { bg: 'rgba(168,85,247,0.12)',  text: '#C084FC',  border: 'rgba(168,85,247,0.25)' },
  'Information Technology':  { bg: 'rgba(59,130,246,0.12)',  text: '#60A5FA',  border: 'rgba(59,130,246,0.25)' },
  'Materials':              { bg: 'rgba(45,212,191,0.12)',  text: '#2DD4BF',  border: 'rgba(45,212,191,0.25)' },
};

export const DEFAULT_SECTOR_COLOR = { bg: 'rgba(148,163,184,0.08)', text: '#64748B', border: 'rgba(148,163,184,0.15)' };

export function getSectorColor(sector: string | null | undefined) {
  if (!sector) return DEFAULT_SECTOR_COLOR;
  return SECTOR_COLORS[sector] || DEFAULT_SECTOR_COLOR;
}

export const PRIMARY_SECTORS = [
  'Technology', 'Industrials', 'Financial Services', 'Consumer Cyclical',
  'Healthcare', 'Energy', 'Consumer Defensive', 'Real Estate',
  'Communication Services', 'Utilities', 'Basic Materials',
] as const;
