/**
 * Copiar al portapapeles con confirmación visual breve. Si el navegador niega el permiso
 * (contexto no seguro, iframe), simplemente no se marca como copiado.
 */
import { useCallback, useEffect, useRef, useState } from "react";

export function useCopy(resetMs = 1400): [boolean, (text: string) => void] {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    },
    [],
  );

  const copy = useCallback(
    (text: string) => {
      navigator.clipboard
        .writeText(text)
        .then(() => {
          setCopied(true);
          if (timer.current !== null) window.clearTimeout(timer.current);
          timer.current = window.setTimeout(() => setCopied(false), resetMs);
        })
        .catch(() => setCopied(false));
    },
    [resetMs],
  );

  return [copied, copy];
}
