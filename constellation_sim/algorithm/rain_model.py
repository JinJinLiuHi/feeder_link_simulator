"""Deterministic rain fade for planning and real-time scheduling."""
from __future__ import annotations
import math

def _rain_rate(station: dict, time_s: float) -> float:
    lat=float(station.get("lat",0));lon=float(station.get("lon",0))
    south=1/(1+math.exp((lat-31)/4.2));east=1/(1+math.exp((103-lon)/5.5))
    base=8+42*south*(.45+.55*east);phase=math.radians((lon*3.7+lat*5.1)%360)
    weather=.58+.27*math.sin(time_s/780+phase)+.15*math.sin(time_s/1930-phase*.7)
    return max(0.,base*max(.05,weather))

def attenuation_db(station: dict,elevation_deg: float,time_s: float) -> float:
    elevation=max(5.,float(elevation_deg));rate=_rain_rate(station,time_s)
    gamma=.0751*rate**1.121
    altitude=max(0.,float(station.get("alt",0))/1000)
    rain_height=max(altitude+.5,5-.075*abs(float(station.get("lat",0))))
    slant=max(.1,(rain_height-altitude)/math.sin(math.radians(elevation)))
    reduction=1/(1+.12*math.sqrt(max(gamma*slant,0)))
    return round(max(0.,gamma*slant*reduction),3)

def link_rain_state(station: dict,elevation_deg: float,time_s: float,enabled: bool,
                    sense_sigma: float,forecast_hours: float,use_p90: bool) -> dict:
    if not enabled:return {"rain_rate_mmh":0.,"actual_attenuation_db":0.,"sensed_attenuation_db":0.,"planning_attenuation_db":0.,"severity":"clear"}
    actual=attenuation_db(station,elevation_deg,time_s)
    phase=math.radians((float(station.get("lat",0))*11+float(station.get("lon",0))*7)%360)
    sensed=max(0.,actual*(1+sense_sigma*math.sin(time_s/37+phase)))
    forecast=attenuation_db(station,elevation_deg,time_s+max(0.,forecast_hours)*3600)
    uncertainty=forecast*min(.65,.10+.08*max(0.,forecast_hours))
    planning=forecast+(1.2816*uncertainty if use_p90 else 0.)
    severity="heavy" if actual>=8 else ("moderate" if actual>=3 else ("light" if actual>=.5 else "clear"))
    return {"rain_rate_mmh":round(_rain_rate(station,time_s),3),"actual_attenuation_db":round(actual,3),"sensed_attenuation_db":round(sensed,3),"planning_attenuation_db":round(planning,3),"severity":severity}
