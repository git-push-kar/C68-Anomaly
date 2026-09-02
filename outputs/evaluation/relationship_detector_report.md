# Relationship Anomaly Detection - Faults 3,9,15

18 defined pairs from KNOWN_PROCESS_RELATIONSHIPS and SUBSYSTEM_GROUPS, no invented

Baselines from 350 normal train, thresholds p99 from 75 normal val (same methodology as 0.687, not same value)

| relationship | normal_event | fault3 | fault9 | fault15 |
|--------------|--------------|--------|--------|---------|
| act_A_C_Feed_Flow->Reactor_Feed_Rate | 0.080 | 0.090 | 0.086 | 0.102 |
| act_A_Feed_Flow->Reactor_Feed_Rate | 0.093 | 0.098 | 0.100 | 0.106 |
| act_Condenser_Cooling_Water_Flow->Separator_Temperature | 0.080 | 0.092 | 0.106 | 0.124 |
| act_Condenser_Cooling_Water_Outlet_Temperature->Separator_Temperature | 0.120 | 0.104 | 0.116 | 0.142 |
| act_D_Feed_Flow->Reactor_Feed_Rate | 0.107 | 0.102 | 0.106 | 0.108 |
| act_Reactor_Cooling_Water_Flow->Reactor_Temperature | 0.093 | 0.096 | 0.102 | 0.086 |
| act_Reactor_Cooling_Water_Outlet_Temperature->Reactor_Temperature | 0.133 | 0.096 | 0.106 | 0.108 |
| act_Reactor_Temperature->Reactor_Pressure | 0.107 | 0.086 | 0.098 | 0.106 |
| act_Separator_Temperature->Separator_Pressure | 0.147 | 0.116 | 0.120 | 0.152 |
| lag_A_C_Feed_Flow->Reactor_Feed_Rate_lag10 | 0.080 | 0.094 | 0.092 | 0.084 |
| lag_A_C_Feed_Flow->Reactor_Feed_Rate_lag5 | 0.120 | 0.070 | 0.072 | 0.084 |
| lag_Condenser_Cooling_Water_Flow->Separator_Temperature_lag10 | 0.093 | 0.070 | 0.072 | 0.094 |
| lag_Condenser_Cooling_Water_Flow->Separator_Temperature_lag5 | 0.067 | 0.074 | 0.076 | 0.082 |
| lag_Reactor_Cooling_Water_Flow->Reactor_Temperature_lag10 | 0.053 | 0.062 | 0.064 | 0.066 |
| lag_Reactor_Cooling_Water_Flow->Reactor_Temperature_lag5 | 0.080 | 0.074 | 0.064 | 0.078 |
| pair_Reactor_Feed_Rate_vs_Reactor_Temperature | 0.093 | 0.066 | 0.068 | 0.070 |
| pair_Reactor_Temperature_vs_Reactor_Pressure | 0.080 | 0.078 | 0.086 | 0.082 |
| pair_Separator_Temperature_vs_Separator_Pressure | 0.133 | 0.084 | 0.058 | 0.118 |
