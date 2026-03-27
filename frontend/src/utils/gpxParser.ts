export interface GpxStats {
  elevationProfile: { dist: number; ele: number }[];
  totalDistanceKm: number;
  elevationGainM: number;
}

function haversineM(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371000;
  const toRad = (x: number) => (x * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function parseGpx(text: string): GpxStats {
  const xml = new DOMParser().parseFromString(text, "text/xml");
  const trkpts = Array.from(xml.querySelectorAll("trkpt, rtept"));

  const raw = trkpts.map((pt) => ({
    lat: parseFloat(pt.getAttribute("lat") ?? "0"),
    lon: parseFloat(pt.getAttribute("lon") ?? "0"),
    ele: parseFloat(pt.querySelector("ele")?.textContent ?? "0"),
  }));

  let totalDist = 0;
  let elevGain = 0;
  const profile: { dist: number; ele: number }[] = [];

  const HYSTERESIS = 5;
  let lastEle = raw[0]?.ele ?? 0;

  for (let i = 0; i < raw.length; i++) {
    if (i > 0) {
      const d = haversineM(raw[i - 1].lat, raw[i - 1].lon, raw[i].lat, raw[i].lon);
      totalDist += d;
      const diff = raw[i].ele - lastEle;
      if (diff >= HYSTERESIS) {
        elevGain += diff;
        lastEle = raw[i].ele;
      } else if (diff <= -HYSTERESIS) {
        lastEle = raw[i].ele;
      }
    }
    profile.push({ dist: totalDist, ele: raw[i].ele });
  }

  // Downsample to max 300 points for SVG rendering
  const MAX = 300;
  const sampled =
    profile.length > MAX
      ? profile.filter((_, i) => i % Math.floor(profile.length / MAX) === 0)
      : profile;

  return {
    elevationProfile: sampled,
    totalDistanceKm: totalDist / 1000,
    elevationGainM: Math.round(elevGain),
  };
}
