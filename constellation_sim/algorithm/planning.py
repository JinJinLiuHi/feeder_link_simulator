"""文档第5.6、5.7和第6章对应的站星规划、实时调度与站点评估。"""
from __future__ import annotations
import csv, heapq, math
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence
from .rain_model import link_rain_state

EARTH_RADIUS_M=6_378_135.0
C_KM_PER_MS=299.792458

@dataclass(frozen=True)
class ModcodRow:
    name:str; minimum_snr_db:float; spectral_efficiency:float; rate_gbps:float|None=None

DEFAULT_MODCOD_TABLE=(ModcodRow("QPSK 1/4",-2.35,.49),ModcodRow("QPSK 1/2",1,1),
 ModcodRow("QPSK 3/4",3,1.5),ModcodRow("8PSK 2/3",6,2),
 ModcodRow("16APSK 3/4",9,3),ModcodRow("32APSK 4/5",13,4))

def load_modcod_table(path):
    if not path:return DEFAULT_MODCOD_TABLE
    rows=[]
    with open(path,"r",encoding="utf-8-sig",newline="") as f:
        for i,r in enumerate(csv.DictReader(f),1):
            name=r.get("modcod") or r.get("name") or r.get("档位") or f"MCS-{i}"
            snr=r.get("minimum_snr_db") or r.get("snr_db") or r.get("最低SNR_dB")
            eta=r.get("spectral_efficiency") or r.get("eta") or r.get("频谱效率")
            rate=r.get("rate_gbps") or r.get("速率_Gbps")
            if snr is None or (eta is None and rate is None):raise ValueError("SNR—速率表字段不完整")
            rows.append(ModcodRow(str(name),float(snr),float(eta or 0),float(rate) if rate not in (None,"") else None))
    if not rows:raise ValueError("SNR—速率表没有有效数据行")
    return tuple(sorted(rows,key=lambda x:x.minimum_snr_db))

@dataclass
class PlannerConfig:
    planning_period_s:float=300.; planning_horizon_s:float=1200.; planning_slot_s:float=15.
    fast_evaluation_period_s:float=1.; fast_adjust_period_s:float=5.; route_update_period_s:float=15.
    freeze_window_s:float=30.; bandwidth_mhz:float=500.; snr_margin_db:float=1.
    capacity_threshold_ratio:float=.85; switching_loss_ratio:float=.02
    modcod_table:Sequence[ModcodRow]=DEFAULT_MODCOD_TABLE
    rain_enabled:bool=True; rain_sense_sigma:float=.20
    rain_forecast_hours:float=2.; rain_use_p90:bool=True
    minimum_dwell_s:float=5.; switching_gain_ratio:float=.03
    traffic_landing_ratio:float=.30

def link_state(elevation,cfg,rain_attenuation_db=0.):
    snr=-1+17*max(0,math.sin(math.radians(elevation)))-max(0.,float(rain_attenuation_db)); selected=None
    for row in cfg.modcod_table:
        if snr-cfg.snr_margin_db>=row.minimum_snr_db:selected=row
    if selected is None:return round(snr,3),"不可用",0.
    rate=selected.rate_gbps if selected.rate_gbps is not None else cfg.bandwidth_mhz*selected.spectral_efficiency/1000
    return round(snr,3),selected.name,round(rate,6)

def _graph(positions,links):
    xyz=[]
    for r in positions:
        lat,lon=map(math.radians,(r[2],r[3])); radius=EARTH_RADIUS_M+r[4]
        xyz.append((radius*math.cos(lat)*math.cos(lon),radius*math.cos(lat)*math.sin(lon),radius*math.sin(lat)))
    g=[[] for _ in positions]
    for a,b,*_ in links:
        d=math.dist(xyz[a],xyz[b]);g[a].append((b,d));g[b].append((a,d))
    return g

def _nearest_demand(g,demands):
    hops=[10**9]*len(g); strength=[0.]*len(g);q=deque()
    for s,d in demands.items():hops[s]=0;strength[s]=d;q.append(s)
    while q:
        n=q.popleft()
        for v,_ in g[n]:
            if hops[v]>hops[n]+1:hops[v]=hops[n]+1;strength[v]=strength[n];q.append(v)
            elif hops[v]==hops[n]+1:strength[v]=max(strength[v],strength[n])
    return hops,strength

def _nearest_demand_distance(g,demands):
    distance=[math.inf]*len(g);queue=[]
    for satellite in demands:distance[satellite]=0.;heapq.heappush(queue,(0.,satellite))
    while queue:
        current,node=heapq.heappop(queue)
        if current!=distance[node]:continue
        for neighbor,edge_distance in g[node]:
            candidate=current+edge_distance
            if candidate<distance[neighbor]:distance[neighbor]=candidate;heapq.heappush(queue,(candidate,neighbor))
    return distance

def _component_demand(g,demands):
    """标出每个ISL连通分量内可汇聚的业务，避免把容量方案投到不可达分量。"""
    values=[0.]*len(g);seen=set()
    for start in range(len(g)):
        if start in seen:continue
        stack=[start];seen.add(start);nodes=[]
        while stack:
            n=stack.pop();nodes.append(n)
            for v,_ in g[n]:
                if v not in seen:seen.add(v);stack.append(v)
        demand=sum(demands.get(n,0.) for n in nodes)
        for n in nodes:values[n]=demand
    return values

def _routes(g,feeders,demands):
    dist=[math.inf]*len(g); parent=[-1]*len(g);root=[-1]*len(g);q=[]
    for s in feeders:dist[s]=0;root[s]=s;heapq.heappush(q,(0,s))
    while q:
        d,n=heapq.heappop(q)
        if d!=dist[n]:continue
        for v,w in g[n]:
            if d+w<dist[v]:dist[v]=d+w;parent[v]=n;root[v]=root[n];heapq.heappush(q,(d+w,v))
    result=[]
    for source,demand in demands.items():
        if root[source]<0:continue
        path=[source];n=source
        while parent[n]>=0 and len(path)<=len(g):n=parent[n];path.append(n)
        result.append({"source":source,"feeder":root[source],"path":path,"hops":len(path)-1,
                       "distance_km":round(dist[source]/1000,3),"isl_propagation_delay_ms":round(dist[source]/1000/C_KM_PER_MS,4),
                       "demand_gbps":round(demand,6)})
    return result

def _gsl_distance_km(altitude_m,elevation_deg):
    radius=EARTH_RADIUS_M;orbit_radius=radius+max(0.,float(altitude_m));elevation=math.radians(float(elevation_deg))
    return (math.sqrt(max(0.,orbit_radius*orbit_radius-(radius*math.cos(elevation))**2))-radius*math.sin(elevation))/1000

def _attach_feeder_delay(routes,links):
    feeder_links={int(x["satellite"]):x for x in links}
    for route in routes:
        link=feeder_links.get(int(route.get("feeder",-1)));gsl_distance=float((link or {}).get("gsl_distance_km",0))
        route["gsl_distance_km"]=round(gsl_distance,3)
        route["gsl_propagation_delay_ms"]=round(gsl_distance/C_KM_PER_MS,4)
        route["propagation_delay_ms"]=round(float(route.get("isl_propagation_delay_ms",0))+gsl_distance/C_KM_PER_MS,4)
    return routes

def _route_metrics(routes):
    if not routes:return 0.,0.,0.,0.,0.
    demand=sum(x["demand_gbps"] for x in routes)
    weighted_delay=sum(float(x.get("propagation_delay_ms",x["distance_km"]/C_KM_PER_MS))*x["demand_gbps"] for x in routes)/max(demand,1e-12)
    ordered=sorted(routes,key=lambda x:float(x.get("propagation_delay_ms",0)));cumulative=0.;p95=0.
    for route in ordered:
        cumulative+=float(route["demand_gbps"])
        if cumulative>=.95*demand:p95=float(route.get("propagation_delay_ms",0));break
    return (round(sum(x["hops"] for x in routes)/len(routes),3),
            round(sum(x["hops"]*x["demand_gbps"] for x in routes)/max(demand,1e-12),3),
            round(weighted_delay,3),round(p95,3),round(max(float(x.get("propagation_delay_ms",0)) for x in routes),3))

@dataclass
class PlanningState:
    generated_at_s:float=-math.inf; generation:int=0; plans:list=field(default_factory=list)
    active_plan_id:str="slow_plan"; last_switch_s:float=-math.inf; switch_count:int=0
    last_scheduled_pairs:set=field(default_factory=set)
    last_evaluation_s:float=-math.inf; last_adjust_s:float=-math.inf
    last_route_update_s:float=-math.inf; last_result:dict|None=None
    execution_plan:dict|None=None

class SlowFastLoopPlanner:
    """慢环生成一套规划骨架，快环逐帧执行和局部修复。"""
    def __init__(self,stations,config=None):self.stations=list(stations);self.config=config or PlannerConfig();self.state=PlanningState()
    def _candidates(self,gsls,t,planning=False):
        out=[]
        for gid,sid,elev in gsls:
            rain=link_rain_state(self.stations[int(gid)],float(elev),t,self.config.rain_enabled,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90)
            fade=rain["planning_attenuation_db"] if planning else rain["sensed_attenuation_db"]
            snr,mcs,rate=link_state(float(elev),self.config,fade)
            if rate>0:out.append({"gateway":int(gid),"satellite":int(sid),"elevation_deg":float(elev),"snr_db":snr,
                                  "modcod":mcs,"rate_gbps":rate,"connection":1,"array_mode":1,"power_factor":1.0,
                                  "rain_attenuation_db":fade,"rain_actual_db":rain["actual_attenuation_db"],
                                  "rain_rate_mmh":rain["rain_rate_mmh"],"rain_severity":rain["severity"]})
        return out
    def _select(self,candidates,g,demands):
        hops,strength=_nearest_demand(g,demands)
        distances=_nearest_demand_distance(g,demands)
        component_demand=_component_demand(g,demands)
        if demands:candidates=[x for x in candidates if component_demand[x["satellite"]]>0]
        # 文档P0/P1/P2：候选在进入此处前已满足可见性等硬约束；随后按
        # 可路由容量、业务加权路径代价和普通路径代价依次排序。
        key=lambda x:(-min(x["rate_gbps"],component_demand[x["satellite"]]),
                      (distances[x["satellite"]]/1000+float(x.get("gsl_distance_km",0)))/max(strength[x["satellite"]],1e-9),
                      distances[x["satellite"]]/1000+float(x.get("gsl_distance_km",0)),hops[x["satellite"]],-x["elevation_deg"])
        remaining={i:max(0,int(s.get("channel_count",s.get("antenna_count",20)))) for i,s in enumerate(self.stations)}
        selected=[];used=set()
        for x in sorted(candidates,key=key):
            if remaining.get(x["gateway"],0)<=0 or x["satellite"] in used:continue
            selected.append(x.copy());remaining[x["gateway"]]-=1;used.add(x["satellite"])
        return selected
    def generate(self,t,positions,links,gsls,satellite_traffic):
        candidates=self._candidates(gsls,t,planning=True);demands={int(x[0]):float(x[1])*self.config.traffic_landing_ratio for x in satellite_traffic if float(x[1])>0};g=_graph(positions,links)
        for x in candidates:x["gsl_distance_km"]=_gsl_distance_km(positions[int(x["satellite"])][4],x["elevation_deg"])
        chosen=self._select(candidates,g,demands);routes=_routes(g,[x["satellite"] for x in chosen],demands) if chosen else []
        _attach_feeder_delay(routes,chosen);avg,bweighted,delay,p95_delay,max_delay=_route_metrics(routes)
        usage=[{"gateway":i,"used":sum(x["gateway"]==i for x in chosen),"available":int(s.get("channel_count",s.get("antenna_count",20)))} for i,s in enumerate(self.stations)]
        plan={"id":"slow_plan","name":"慢环站星连接规划","generated_at_s":t,
          "valid_from_s":t,"valid_until_s":t+self.config.planning_horizon_s,
          "slot_s":self.config.planning_slot_s,"slot_count":max(1,math.ceil(self.config.planning_horizon_s/self.config.planning_slot_s)),
          "freeze_until_s":t+self.config.freeze_window_s,"links":chosen,"routes":routes,
          "total_capacity_gbps":round(sum(x["rate_gbps"] for x in chosen),6),"total_demand_gbps":round(sum(demands.values()),6),
          "traffic_satellite_count":len(demands),
          "reachable_demand_gbps":round(sum(x["demand_gbps"] for x in routes),6),"gateway_count":len({x["gateway"] for x in chosen}),
          "average_hops":avg,"business_weighted_hops":bweighted,"business_propagation_delay_ms":delay,
          "p95_propagation_delay_ms":p95_delay,"maximum_propagation_delay_ms":max_delay,"latency_scope":"服务卫星至信关站（ISL+馈电GSL）","station_usage":usage}
        plans=[plan]
        self.state.generated_at_s=t;self.state.generation+=1;self.state.plans=plans;self.state.execution_plan=plan
        self.state.last_route_update_s=t;return plans

    def _fast_adjust(self,t,plan,positions,links,gsls,satellite_traffic):
        """固定慢环资源边界，在当前可见候选中替换失效馈电连接。"""
        candidates=self._candidates(gsls,t,planning=False);g=_graph(positions,links)
        for x in candidates:x["gsl_distance_km"]=_gsl_distance_km(positions[int(x["satellite"])][4],x["elevation_deg"])
        demands={int(x[0]):float(x[1])*self.config.traffic_landing_ratio for x in satellite_traffic if float(x[1])>0}
        ranked=self._select(candidates,g,demands)
        # Fast loop keeps the slow-loop resource envelope: each gateway may use
        # only the number of antenna slots allocated by the current plan.  It
        # may replace a degraded visible link with a better ranked candidate.
        quota={i:sum(x["gateway"]==i for x in plan["links"]) for i in range(len(self.stations))}
        old_pairs={(x["gateway"],x["satellite"]) for x in plan["links"]}
        chosen=[];used=set();remaining=dict(quota)
        for now in ranked:
            if remaining.get(now["gateway"],0)>0 and now["satellite"] not in used:
                now=now.copy();now["is_substitute"]=(now["gateway"],now["satellite"]) not in old_pairs
                chosen.append(now);used.add(now["satellite"]);remaining[now["gateway"]]-=1
        routes=_routes(g,[x["satellite"] for x in chosen],demands) if chosen else []
        _attach_feeder_delay(routes,chosen)
        adjusted=plan.copy();adjusted["links"]=chosen;adjusted["routes"]=routes
        adjusted["total_capacity_gbps"]=round(sum(x["rate_gbps"] for x in chosen),6)
        return adjusted
    def _evaluate(self,t,plan,pairs,total_demand,edges):
        active=[];ideal=sum(x["rate_gbps"] for x in plan["links"]);actual=0
        for old in plan["links"]:
            elev=pairs.get((old["gateway"],old["satellite"]))
            if elev is None:continue
            rain=link_rain_state(self.stations[int(old["gateway"])],elev,t,self.config.rain_enabled,self.config.rain_sense_sigma,self.config.rain_forecast_hours,self.config.rain_use_p90)
            snr,mcs,rate=link_state(elev,self.config,rain["sensed_attenuation_db"])
            if rate<=0:continue
            x=old.copy();x.update(elevation_deg=elev,snr_db=snr,modcod=mcs,rate_gbps=rate,rain_attenuation_db=rain["sensed_attenuation_db"],rain_actual_db=rain["actual_attenuation_db"],rain_rate_mmh=rain["rain_rate_mmh"],rain_severity=rain["severity"]);active.append(x);actual+=rate
        active_feeders={x["satellite"] for x in active}
        valid=[r for r in plan["routes"] if r.get("feeder") in active_feeders and all((min(a,b),max(a,b)) in edges for a,b in zip(r["path"],r["path"][1:]))]
        reachable=sum(x["demand_gbps"] for x in valid)
        if plan["routes"]:
            demand_by_feeder={}
            for route in valid:
                feeder=int(route.get("feeder",-1));demand_by_feeder[feeder]=demand_by_feeder.get(feeder,0.)+float(route.get("demand_gbps",0))
            for x in active:
                x["scheduled_gbps"]=round(min(float(x["rate_gbps"]),demand_by_feeder.get(int(x["satellite"]),0.)),6)
                x["utilization"]=round(x["scheduled_gbps"]/max(x["rate_gbps"],1e-12),6)
            scheduled=sum(float(x["scheduled_gbps"]) for x in active)
        else:
            scheduled=min(total_demand,actual)
            for x in active:
                x["scheduled_gbps"]=round(scheduled*x["rate_gbps"]/actual,6) if actual else 0.
                x["utilization"]=round(x["scheduled_gbps"]/max(x["rate_gbps"],1e-12),6)
        avg,bweighted,delay,p95_delay,max_delay=_route_metrics(valid)
        return {"id":plan["id"],"name":plan["name"],"ideal_capacity_gbps":round(ideal,6),"actual_capacity_gbps":round(actual,6),
          "capacity_achievement_ratio":round(actual/max(ideal,1e-12),6),"scheduled_gbps":round(scheduled,6),
          "unserved_gbps":round(max(0,total_demand-scheduled),6),"active_links":active,"planned_link_count":len(plan["links"]),
          "active_link_count":len(active),"route_available_ratio":round(1.0 if not plan.get("traffic_satellite_count",0) else len(valid)/plan["traffic_satellite_count"],6),
          "average_hops":avg,"business_weighted_hops":bweighted,"business_propagation_delay_ms":delay,
          "p95_propagation_delay_ms":p95_delay,"maximum_propagation_delay_ms":max_delay,"latency_scope":"服务卫星至信关站（ISL+馈电GSL）",
          "route_count":len(valid),"routes":valid}
    def update(self,t,positions,links,gsls,satellite_traffic):
        first_plan=not self.state.plans;due=first_plan or t<self.state.generated_at_s or t-self.state.generated_at_s>=self.config.planning_period_s
        planning_reason="首次生成慢周期资源方案" if first_plan else ("到达滚动规划周期" if due else "未到规划周期，沿用当前方案")
        if due:self.generate(t,positions,links,gsls,satellite_traffic)
        evaluation_due=due or self.state.last_result is None or t<self.state.last_evaluation_s or t-self.state.last_evaluation_s>=self.config.fast_evaluation_period_s
        if not evaluation_due:
            result=dict(self.state.last_result);result["planning_event"]=False;result["scheduling_event"]=False;result["planning_reason"]="未到规划周期，沿用当前方案";result["connection_change_count"]=0
            return result
        pairs={(int(g),int(s)):float(e) for g,s,e in gsls};edges={(min(int(a),int(b)),max(int(a),int(b))) for a,b,*_ in links};access_demand=sum(float(x[1]) for x in satellite_traffic);demand=access_demand*self.config.traffic_landing_ratio
        slow_plan=self.state.plans[0];plan=self.state.execution_plan or slow_plan
        active=self._evaluate(t,plan,pairs,demand,edges);action="按慢环规划执行"
        route_due=not due and t-self.state.last_route_update_s>=self.config.route_update_period_s
        if route_due:
            graph=_graph(positions,links);demands={int(x[0]):float(x[1])*self.config.traffic_landing_ratio for x in satellite_traffic if float(x[1])>0}
            plan=plan.copy();plan["routes"]=_routes(graph,[x["satellite"] for x in plan["links"]],demands) if plan["links"] else []
            _attach_feeder_delay(plan["routes"],plan["links"])
            self.state.execution_plan=plan;active=self._evaluate(t,plan,pairs,demand,edges);self.state.last_route_update_s=t
            action="周期更新ISL汇聚路由，保持慢环资源方案"
        threshold=self.config.capacity_threshold_ratio
        demand_ratio=active["unserved_gbps"]/max(demand,1e-12) if active else 1.
        severe=active is None or active["capacity_achievement_ratio"]<.70 or active["route_available_ratio"]<.90 or demand_ratio>.05
        if not due and severe and t-self.state.generated_at_s>=self.config.freeze_window_s:
            self.generate(t,positions,links,gsls,satellite_traffic);slow_plan=self.state.plans[0];plan=self.state.execution_plan or slow_plan
            active=self._evaluate(t,plan,pairs,demand,edges);due=True;planning_reason="容量、路径或未送达业务越限，提前重规划";action="输入越限，提前触发慢环重规划"
        if active and (active["capacity_achievement_ratio"]<threshold or active["route_available_ratio"]<threshold):
            emergency=active["capacity_achievement_ratio"]<.50 or active["route_available_ratio"]<.80
            adjust_due=t-self.state.last_adjust_s>=self.config.fast_adjust_period_s
            if emergency or adjust_due:
                adjusted_plan=self._fast_adjust(t,slow_plan,positions,links,gsls,satellite_traffic)
                adjusted=self._evaluate(t,adjusted_plan,pairs,demand,edges)
                gain_required=active["scheduled_gbps"]*(1+self.config.switching_gain_ratio)
                dwell_ok=(t-self.state.last_switch_s)>=self.config.minimum_dwell_s
                if adjusted["scheduled_gbps"]>=gain_required and dwell_ok:
                    active=adjusted;plan=adjusted_plan;self.state.execution_plan=adjusted_plan
                    action="低于门限，快环局部切换天线指向";self.state.last_adjust_s=t;self.state.last_route_update_s=t
                else:action="低于门限，切换收益或保持时间不足，维持当前连接"
            else:action="低于门限，等待快环最短调整间隔"
        pairs_now={(x["gateway"],x["satellite"]) for x in (active["active_links"] if active else [])}
        change_count=len(pairs_now.symmetric_difference(self.state.last_scheduled_pairs)) if self.state.last_scheduled_pairs else 0;changed=change_count>0
        if changed:self.state.switch_count+=change_count;self.state.last_switch_s=t
        self.state.last_scheduled_pairs=pairs_now
        loss=active["scheduled_gbps"]*self.config.switching_loss_ratio if active and changed else 0
        deployment=self._deployment(active,gsls)
        route_ratio=float(active.get("route_available_ratio",0)) if active else 0.
        capacity_ratio=float(active.get("capacity_achievement_ratio",0)) if active else 0.
        unserved=float(active.get("unserved_gbps",demand)) if active else demand
        rain_affected=sum(float(x.get("rain_actual_db",0))>=3 for x in (active or {}).get("active_links",[]))
        if route_ratio<.999:constraint_source="拓扑可达性"
        elif capacity_ratio<.999 and rain_affected:constraint_source="雨衰与星地链路容量"
        elif capacity_ratio<.999:constraint_source="星地链路容量"
        elif unserved>.001:constraint_source="业务需求与资源容量"
        else:constraint_source="无显著约束"
        decision_context={
          "access_traffic_gbps":round(access_demand,6),"landing_ratio":round(self.config.traffic_landing_ratio,6),
          "traffic_demand_gbps":round(demand,6),"traffic_satellite_count":len(satellite_traffic),
          "isl_count":len(links),"gsl_candidate_count":len(gsls),
          "visible_feeder_satellite_count":len({int(x[1]) for x in gsls}),
          "gateway_count":len(self.stations),"planned_link_count":len(plan.get("links",[])),
          "active_link_count":int(active.get("active_link_count",0)) if active else 0,
          "route_available_ratio":round(route_ratio,6),"capacity_achievement_ratio":round(capacity_ratio,6),
          "constraint_source":constraint_source}
        result={"generation":self.state.generation,"generated_at_s":self.state.generated_at_s,"planning_period_s":self.config.planning_period_s,
          "planning_horizon_s":self.config.planning_horizon_s,"planning_slot_s":self.config.planning_slot_s,
          "fast_evaluation_period_s":self.config.fast_evaluation_period_s,"fast_adjust_period_s":self.config.fast_adjust_period_s,
          "route_update_period_s":self.config.route_update_period_s,"freeze_window_s":self.config.freeze_window_s,
          "capacity_threshold_ratio":threshold,"active_plan_id":self.state.active_plan_id,"last_switch_s":self.state.last_switch_s,
          "switch_count":self.state.switch_count,"switching_loss_gbps":round(loss,6),"delivered_gbps":round(max(0,(active["scheduled_gbps"] if active else 0)-loss),6),
          "action":action,"planning_reason":planning_reason,"connection_change_count":change_count,"planning_event":due,"scheduling_event":True,
          "rain_summary":self._rain_summary(active),
          "next_planning_time_s":round(self.state.generated_at_s+self.config.planning_period_s,6),
          "plans":self.state.plans,"evaluations":[active],"active_plan":slow_plan,"execution_plan":plan,
          "active_evaluation":active,"gateway_deployment":deployment,"decision_context":decision_context}
        self.state.last_evaluation_s=t;self.state.last_result=result
        return result
    def _rain_summary(self,active):
        rows=(active or {}).get("active_links",[])
        values=[float(x.get("rain_actual_db",0)) for x in rows]
        return {"enabled":self.config.rain_enabled,"active_link_count":len(values),
          "affected_link_count":sum(x>=3 for x in values),
          "average_attenuation_db":round(sum(values)/len(values),3) if values else 0.,
          "maximum_attenuation_db":round(max(values),3) if values else 0.}
    def _deployment(self,active,gsls):
        rows=active["active_links"] if active else [];stations=[]
        for gid,s in enumerate(self.stations):
            channels=int(s.get("channel_count",s.get("antenna_count",20)));used=sum(x["gateway"]==gid for x in rows)
            stations.append({"gateway":gid,"name":s.get("name",f"站{gid+1}"),"visible_satellites":sum(int(x[0])==gid for x in gsls),
             "planned_channels":channels,"used_channels":used,"recommended_antennas":max(1,min(channels,used+(1 if used else 0))),"minimum_connection_satisfied":used>0})
        return {"status":"已规划信关站的当前资源状态","station_count":len(stations),
         "active_station_count":sum(x["used_channels"]>0 for x in stations),"minimum_connection_satisfied":all(x["minimum_connection_satisfied"] for x in stations) if stations else False,"stations":stations}
