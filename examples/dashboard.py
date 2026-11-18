"""Run: streamlit run examples/dashboard.py (from the repository root)."""
from pathlib import Path

import streamlit as st
import torch

from physcast_yaw import GaussianWake, Wind, load_config
from physcast_yaw.control import ControllerSettings, optimize_yaw
from physcast_yaw.learning import WakePredictor
from physcast_yaw.physics import TEACHER_PARAMETERS
from physcast_yaw.plots import wake_comparison

torch.set_num_threads(min(4, torch.get_num_threads()))
st.set_page_config(page_title='PhysCast-Yaw', page_icon='🌀', layout='wide')
st.title('PhysCast-Yaw')
st.caption('Wake prediction · constrained yaw control · reproducible comparison')

with st.sidebar:
    st.header('Scene & controller')
    with st.form('controls'):
        config = st.selectbox('Layout', ['configs/three_turbines.json', 'configs/six_turbines.json'])
        speed = st.slider('Ambient speed (m/s)', 4., 15., 8., .25)
        direction = st.slider('Wind FROM (°)', 240., 300., 270., 1.)
        ti = st.slider('Turbulence intensity', .04, .20, .08, .01)
        uncertainty = st.slider('Direction standard deviation (°)', 0., 6., 2., .5)
        checkpoint = st.text_input('Checkpoint', 'outputs/demo/model/model.pt')
        submitted = st.form_submit_button('Compare yaw commands', type='primary')
    st.info('Train once with: physcast-yaw demo --out outputs/demo')

if submitted:
    farm, _, _ = load_config(config)
    wind = Wind(speed, direction, ti)
    settings = ControllerSettings(direction_std_deg=uncertainty, candidates=48, sweeps=2)
    if not Path(checkpoint).is_file():
        st.warning('Checkpoint not found. Run the demo first, then select its model.pt.')
    else:
        with st.spinner('Evaluating feasible yaw actions…'):
            model = WakePredictor(checkpoint)
            commands = {'aligned': dict(yaw_deg=[0.]*farm.n, source='alignment')}
            commands['physics'] = optimize_yaw(farm, wind, settings=settings)
            commands['ai'] = optimize_yaw(farm, wind, model, settings=settings)
            teacher = GaussianWake(farm, TEACHER_PARAMETERS)
            for command in commands.values():
                command['evaluated_power_kw'] = float(teacher.simulate(wind, command['yaw_deg'])['power_kw'].sum())
        st.session_state['comparison'] = (farm, wind, commands)

if 'comparison' in st.session_state:
    farm, wind, commands = st.session_state['comparison']
    columns = st.columns(3)
    baseline = commands['aligned']['evaluated_power_kw']
    for column, method, label in zip(columns, ('aligned', 'physics', 'ai'), ('Wind alignment', 'Physics controller', 'AI controller')):
        command = commands[method]
        gain = 100*(command['evaluated_power_kw']/baseline-1) if baseline > 0 else 0
        column.metric(label, f"{command['evaluated_power_kw']/1000:.3f} MW", f'{gain:+.2f}% vs alignment')
    st.caption(f"AI source: {commands['ai']['source']} · {commands['ai'].get('reason', '')}")
    st.pyplot(wake_comparison(farm, wind, commands))
    st.dataframe([dict(method=key, turbine=f'T{i+1}', yaw_deg=round(y, 2))
                  for key, c in commands.items() for i, y in enumerate(c['yaw_deg'])], hide_index=True)
    table = 'method,turbine,yaw_deg,evaluated_farm_power_kw\n'+''.join(
        f"{key},T{i+1},{y:.6f},{c['evaluated_power_kw']:.6f}\n" for key, c in commands.items() for i,y in enumerate(c['yaw_deg']))
    st.download_button('Download commands and power (CSV)', table, 'yaw-comparison.csv', 'text/csv')
else:
    st.markdown('Choose a wind condition, then **Compare yaw commands**. Each controller sees the same layout and wind.')
st.caption('The maps and displayed power use a synthetic Gaussian teacher. Neural predictions are rotor-effective speeds and power; this view is not field validation.')
