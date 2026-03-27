interface Point {
  dist: number;
  ele: number;
}

interface Props {
  profile: Point[];
}

export default function ElevationProfile({ profile }: Props) {
  if (profile.length < 2) return null;

  const W = 360;
  const H = 100;
  const pad = 4;

  const minEle = Math.min(...profile.map((p) => p.ele));
  const maxEle = Math.max(...profile.map((p) => p.ele));
  const maxDist = profile[profile.length - 1].dist;

  const toX = (dist: number) => pad + (dist / maxDist) * (W - 2 * pad);
  const toY = (ele: number) =>
    H - pad - ((ele - minEle) / (maxEle - minEle || 1)) * (H - 2 * pad);

  const pts = profile.map((p) => `${toX(p.dist)},${toY(p.ele)}`).join(" ");
  const area = `${toX(0)},${H} ` + pts + ` ${toX(maxDist)},${H}`;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="elevationSvg" preserveAspectRatio="none">
      <defs>
        <linearGradient id="eleGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#4a90e2" stopOpacity="0.6" />
          <stop offset="100%" stopColor="#4a90e2" stopOpacity="0.1" />
        </linearGradient>
      </defs>
      <polygon points={area} fill="url(#eleGrad)" />
      <polyline points={pts} fill="none" stroke="#4a90e2" strokeWidth="1.5" />
    </svg>
  );
}
