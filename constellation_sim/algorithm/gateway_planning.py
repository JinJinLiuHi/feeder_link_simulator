"""Offline resource planning for preconfigured gateway stations."""
from __future__ import annotations

import math

EARTH_RADIUS_M = 6_378_135.0


def _distance_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    value = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 2*6371.0*math.asin(min(1.0, math.sqrt(value)))


def plan_gateway_resources(stations, orbit_cache, min_elevation_deg=30.0,
                           station_capacity_gbps=80.0, demand_gbps=1000.0,
                           max_gsl_km=0.0, traffic_cells=None):
    """Plan resources for every preconfigured station without selecting sites."""
    sites=[dict(x) for x in stations]
    if not sites:
        return {"status":"未配置信关站","algorithm":"固定站点资源规划","stations":[],"planned_stations":[]}
    import numpy as np
    frame_count=len(orbit_cache) if orbit_cache is not None else 0
    sample_count=min(61,frame_count)
    sample_indices=sorted({round(i*(frame_count-1)/max(1,sample_count-1)) for i in range(sample_count)}) if frame_count else []
    weighted_cells=[x for x in (traffic_cells or []) if float(getattr(x,"base_weight",0))>0]
    weight_total=sum(float(x.base_weight) for x in weighted_cells)
    metrics=[]
    for index,site in enumerate(sites):
        counts=[]
        for frame_index in sample_indices:
            frame=orbit_cache[frame_index];lat=math.radians(float(site["lat"]));lon=math.radians(float(site["lon"]))
            slat=np.radians(frame[:,0]);slon=np.radians(frame[:,1]);radius=EARTH_RADIUS_M+np.maximum(0.0,frame[:,2])
            cosine=np.clip(math.sin(lat)*np.sin(slat)+math.cos(lat)*np.cos(slat)*np.cos(slon-lon),-1.0,1.0);central=np.arccos(cosine)
            elevation=np.degrees(np.arctan2(radius*np.cos(central)-EARTH_RADIUS_M,radius*np.sin(central)))
            slant=np.sqrt(radius*radius+EARTH_RADIUS_M*EARTH_RADIUS_M-2*radius*EARTH_RADIUS_M*np.cos(central))/1000.0
            valid=elevation>=min_elevation_deg
            if max_gsl_km and max_gsl_km>0:valid&=slant<=max_gsl_km
            counts.append(int(np.count_nonzero(valid)))
        availability=sum(value>0 for value in counts)/max(1,len(counts))
        affinity=(sum(float(x.base_weight)*math.exp(-_distance_km(site,{"lat":x.latitude_deg,"lon":x.longitude_deg})/1800.0) for x in weighted_cells)/weight_total) if weight_total else 1.0
        metrics.append({"gateway":index,"name":site.get("name",f"信关站{index+1}"),"lat":float(site["lat"]),"lon":float(site["lon"]),
                        "availability_ratio":round(availability,6),"average_visible_satellites":round(sum(counts)/max(1,len(counts)),2),
                        "minimum_visible_satellites":min(counts) if counts else 0,"maximum_visible_satellites":max(counts) if counts else 0,
                        "business_affinity":round(affinity,6)})
    raw=[max(1e-6,m["availability_ratio"])*max(.05,m["business_affinity"]) for m in metrics];raw_total=sum(raw);planned=[]
    for index,(site,metric) in enumerate(zip(sites,metrics)):
        share=raw[index]/raw_total if raw_total else 1/len(sites);assigned=demand_gbps*share
        antenna_limit=max(1,int(site.get("antenna_count",20)));channel_limit=max(1,int(site.get("channel_count",antenna_limit)))
        channel_capacity=station_capacity_gbps/max(1,channel_limit);planned_channels=min(channel_limit,max(1,math.ceil(assigned/max(channel_capacity,1e-9))))
        planned_antennas=min(antenna_limit,planned_channels);designed=min(station_capacity_gbps,planned_channels*channel_capacity)
        alternatives=[j for j in range(len(sites)) if j!=index]
        backup=max(alternatives,key=lambda j:(_distance_km(site,sites[j]),metrics[j]["availability_ratio"])) if alternatives else None
        station=dict(site);station.update(configured_antenna_count=antenna_limit,configured_channel_count=channel_limit,
            antenna_count=planned_antennas,channel_count=planned_channels,planning_role="固定部署站")
        planned.append(station);metric.update(planned=True,traffic_share=round(share,6),assigned_demand_gbps=round(assigned,3),
            planned_antennas=planned_antennas,planned_channels=planned_channels,designed_capacity_gbps=round(designed,3),
            backup_gateway=backup,backup_name=sites[backup].get("name",f"信关站{backup+1}") if backup is not None else "--")
    designed_total=sum(x["designed_capacity_gbps"] for x in metrics)
    return {"status":"已生成固定信关站资源规划" if designed_total>=demand_gbps else "已生成资源规划，但配置容量存在缺口",
            "algorithm":"固定站点的时域可见性、业务分担与资源容量规划","sampled_frames":len(sample_indices),
            "station_count":len(sites),"target_demand_gbps":round(float(demand_gbps),3),"designed_capacity_gbps":round(designed_total,3),
            "capacity_gap_gbps":round(max(0.0,demand_gbps-designed_total),3),"capacity_satisfied":designed_total>=demand_gbps,
            "stations":metrics,"planned_stations":planned}


def plan_gateway_automatic(orbit_cache, min_elevation_deg=30.0, station_capacity_gbps=80.0,
                           demand_gbps=1000.0, max_gsl_km=0.0, traffic_cells=None,
                           maximum_sites=12, redundancy=2, antenna_limit=20):
    """Create an internal China search grid and select a deployment automatically."""
    # These coordinates represent city-level planning anchors rather than exact
    # construction parcels.  Traffic-cell centroids are deliberately excluded:
    # without a DEM/water/land-use mask they could fall on rivers or steep terrain.
    anchors=(("佳木斯",46.80,130.32),("北京",39.90,116.40),("喀什",39.47,75.99),("拉萨",29.65,91.17),("三亚",18.25,109.51),
             ("上海",31.23,121.47),("广州",23.13,113.26),("成都",30.57,104.07),("西安",34.34,108.94),
             ("乌鲁木齐",43.83,87.62),("哈尔滨",45.80,126.53),("沈阳",41.81,123.43),("呼和浩特",40.84,111.75),
             ("济南",36.65,117.12),("南京",32.06,118.80),("武汉",30.59,114.30),("重庆",29.56,106.55),
             ("昆明",25.04,102.71),("兰州",36.06,103.83),("郑州",34.75,113.62),("南宁",22.82,108.32),
             ("福州",26.08,119.30),("长沙",28.23,112.94),("合肥",31.82,117.23),("银川",38.49,106.23))
    candidates=[{"name":name,"lat":lat,"lon":lon,"alt":0,"antenna_count":int(antenna_limit),"channel_count":int(antenna_limit),"site_basis":"城市级基础设施规划锚点"} for name,lat,lon in anchors]
    evaluated=plan_gateway_resources(candidates,orbit_cache,min_elevation_deg,station_capacity_gbps,demand_gbps,max_gsl_km,traffic_cells)
    metrics=evaluated["stations"];selected=[];remaining=set(range(len(candidates)))
    required=max(int(redundancy),math.ceil(demand_gbps/max(station_capacity_gbps,1e-9)))
    target=min(max(1,int(maximum_sites)),max(1,required),len(candidates))
    while remaining and len(selected)<target:
        best=None
        for i in remaining:
            diversity=1.0 if not selected else min(1.0,min(_distance_km(candidates[i],candidates[j]) for j in selected)/1800.0)
            score=.45*metrics[i]["availability_ratio"]+.30*metrics[i]["business_affinity"]+.25*diversity
            value=(score,metrics[i]["minimum_visible_satellites"],-i)
            if best is None or value>best[0]:best=(value,i)
        chosen=best[1];selected.append(chosen);remaining.remove(chosen)
    selected_sites=[candidates[i] for i in selected]
    result=plan_gateway_resources(selected_sites,orbit_cache,min_elevation_deg,station_capacity_gbps,demand_gbps,max_gsl_km,traffic_cells)
    result.update(mode="automatic",algorithm="城市级基础设施锚点 + 时域可见性 + 业务需求 + 地域分散自动规划",
                  search_point_count=len(candidates),required_station_count=required,maximum_sites=int(maximum_sites),redundancy=int(redundancy),
                  station_count_rule=f"max(冗余{int(redundancy)}站, ceil(峰值落地业务{demand_gbps:.2f}/单站无线规划能力{station_capacity_gbps:.2f}))={required}站，受上限{int(maximum_sites)}站约束",
                  limitations="当前输出为城市级规划位置，并非精确建设坐标；精确选址仍需接入DEM坡度、水体、土地利用、实测雨强、电力和骨干网数据")
    selected_original=[selected_sites[i].get("name","") for i in range(len(selected_sites))]
    for order,station in enumerate(result["planned_stations"]):
        original=selected_original[order];station["name"]=f"GW-A{order+1:02d} {original}"
    for order,row in enumerate(result["stations"]):
        original=selected_original[order];row["name"]=f"GW-A{order+1:02d} {original}"
        row["selection_reason"]=f"{selected_sites[order].get('site_basis','内部搜索点')}；可用率{100*row.get('availability_ratio',0):.1f}%；业务权重{100*row.get('business_affinity',0):.1f}%"
        if row.get("backup_gateway") is not None:row["backup_name"]=f"GW-A{int(row['backup_gateway'])+1:02d}"
    return result
