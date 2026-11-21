/** Marca de DocuERP: hoja con pliegue y dos líneas de cita en esmeralda. */
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden>
      <defs>
        <linearGradient id="docuerp-g" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse">
          <stop stopColor="#1B2230" />
          <stop offset="1" stopColor="#0E1218" />
        </linearGradient>
      </defs>
      <rect width="32" height="32" rx="8.5" fill="url(#docuerp-g)" />
      <rect x="0.5" y="0.5" width="31" height="31" rx="8" stroke="#10B981" strokeOpacity="0.42" />
      <path d="M10 8.5h7.2L22 13.3V23.5H10z" stroke="#E7EAEE" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M17 8.5v5h5" stroke="#E7EAEE" strokeWidth="1.5" strokeLinejoin="round" />
      <path d="M13 17.5h6M13 20.5h3.5" stroke="#34D399" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );
}
