import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from threading import Event
import os





app = dash.Dash(__name__,prevent_initial_callbacks='initial_duplicate')
server=app.server


def can_see(r, vision):
    # input r is an n-by-3 position matrix, where n is the number of birds
    # delta_r is an n-by-n matrix where delta_r[i,j] is the distance between bird_i and bird_j
    # can_see is an n-by-n matrix where True if in vision False if out of vision
    x = r[:,0].reshape(1,-1)
    y = r[:,1].reshape(1,-1)
    z = r[:,2].reshape(1,-1)
    delta_x = x-x.T
    delta_y = y-y.T
    delta_z = z-z.T
    delta_r = np.stack((delta_x,delta_y, delta_z))
    delta_r = np.sqrt( np.sum(delta_r**2,axis=0) )
    in_vision = np.full(delta_r.shape, True)
    # we have a vision parameter that will exclude neighbors farther than a certain threshold
    if vision != 0:
        in_vision[delta_r > vision]=False
    return in_vision

def scale_factor(r):  
    # the force of one bird acting on another will by default scale inversely proportional to the distance between them squared.  
    # the output of this is an n-by-n matrix where scale[m,n] = 1/|r(m)-r(n)|^2
    x = r[:,0].reshape(1,-1) 
    y = r[:,1].reshape(1,-1)
    z = r[:,2].reshape(1,-1)
    delta_x = x-x.T
    delta_y = y-y.T
    delta_z = z-z.T
    delta_r = np.stack((delta_x,delta_y,delta_z))
    delta_r2 = np.sum(delta_r**2,axis=0)
    scale = np.zeros_like(delta_r2,dtype=float)
    scale[delta_r2!=0] =1/delta_r2[delta_r2!=0] 
    return scale

def F_avoid(r,vision, avoid_factor):
    in_vision = can_see(r,vision)
    scale = scale_factor(r)
    fudge_factor = 7.5 
    n_in_vision = np.sum(in_vision,axis=1).astype(float)
    mask = n_in_vision!=0
    n_in_vision[mask] = 1/n_in_vision[mask]
    x = r[:,0].reshape(1,-1)
    y = r[:,1].reshape(1,-1)
    z = r[:,2].reshape(1,-1)
    delta_x = -(x-x.T) * scale * in_vision
    delta_y = -(y-y.T) * scale * in_vision
    delta_z = -(z-z.T) * scale * in_vision
    dx = (np.sum(delta_x, axis=1)) 
    dy = (np.sum(delta_y, axis=1)) 
    dz = (np.sum(delta_z, axis=1))         
    dv = np.stack((dx,dy,dz),axis=1)*n_in_vision.reshape(-1,1)*avoid_factor*fudge_factor
    return dv
    


def F_align(r,v,vision,match_factor):  
    in_vision = can_see(r,vision)
    scale = scale_factor(r)
    fudge_factor = 13.3
    n_in_vision = np.sum(in_vision,axis=1).astype(float)
    mask = n_in_vision!=0
    n_in_vision[mask] = 1/n_in_vision[mask]
    vx = v[:,0]
    vy = v[:,1]
    vz = v[:,2]
    n=vx.size
    vx_block = (np.tile(vx,(n,1))) * scale * in_vision
    vy_block = (np.tile(vy,(n,1))) * scale * in_vision
    vz_block = (np.tile(vz,(n,1))) * scale * in_vision
    dvx = (np.sum(vx_block, axis=1))
    dvy = (np.sum(vy_block, axis=1))
    dvz = (np.sum(vz_block, axis=1))
    dv = np.stack((dvx,dvy,dvz), axis=1)*np.sqrt(n_in_vision).reshape(-1,1)*match_factor*fudge_factor
    return dv
    
def F_cohesion(r,vision,centering_factor):
    in_vision = can_see(r,vision)
    x = r[:,0]
    y = r[:,1]
    z = r[:,2]
    n = x.size
    fudge_factor = 0.025
    if n<=1:
        dv = np.array([0,0,0])
    else:
        in_vision = can_see(r,vision)
        x_block = (np.tile(x,(n,1))) * in_vision
        y_block = (np.tile(y,(n,1))) * in_vision
        z_block = (np.tile(z,(n,1))) * in_vision
        np.fill_diagonal(x_block,0)
        np.fill_diagonal(y_block,0)
        np.fill_diagonal(z_block,0)
        x_ave = np.sum(x_block, axis=1)/(np.max([1,n-1]))
        y_ave = np.sum(y_block, axis=1)/(np.max([1,n-1]))
        z_ave = np.sum(z_block, axis=1)/(np.max([1,n-1]))
        dx =np.log(1+np.abs(x_ave-x))*np.sign(x_ave-x)  
        dy =np.log(1+np.abs(y_ave-y))*np.sign(y_ave-y)
        dz =np.log(1+np.abs(z_ave-z))*np.sign(z_ave-z)  
        dv = np.stack((dx,dy,dz),axis=1)*centering_factor*fudge_factor
    return dv

def mind_edges(r,L,margin, turn_factor):
    # The birds will be constrained to a cube of center (0,0,0) and side length 2L, turn around when approach margin.
    fudge_factor = 1
    x=r[:,0]
    y=r[:,1]
    z=r[:,2]
    top = L -margin
    bot = top*-1
    n= len(x)
    dx = np.zeros(x.size)
    dy = np.zeros(y.size)
    dz = np.zeros(z.size)
    dx[x<bot] = (bot-x[x<bot])
    dx[x>top] = (top-x[x>top])
    dy[y<bot] = (bot-y[y<bot])
    dy[y>top] = (top-y[y>top])
    dz[z<bot] = (bot-z[z<bot])
    dz[z>top] = (top-z[z>top])
    # the other forces on the birds tend to scale with n, so let's scale this with n, too
    dv = (np.stack((dx,dy,dz), axis=1)) * n *turn_factor*fudge_factor 
    return dv
    
def limit_speed(v,min_speed,max_speed):
    v_norm = np.linalg.norm(v,axis=1)
    v_hat = np.zeros(v.shape)
    v_hat[v_norm !=0] = v/v_norm[:,np.newaxis]
    v_norm[v_norm < min_speed] = min_speed
    v_norm[v_norm > max_speed] = max_speed
    v = v_hat*v_norm[:,np.newaxis]
    return v

def initialize_birds(n, L, margin, min_speed, max_speed):
    mean = [0,0,0]
    mean_v = np.random.uniform(-max_speed, max_speed, 3)
    edge = L - margin 
    cov = np.identity(3)*edge 
    r = np.random.multivariate_normal(mean, cov,n)
    cov_v = np.identity(3)*0.2 
    v =  np.random.multivariate_normal(mean_v, cov_v, n)
    v_hat = v/np.linalg.norm(v,axis=1).reshape(-1,1)
    v_mag = np.random.uniform(min_speed , max_speed, (n,1))
    v = v_hat*v_mag
    return r,v


def update_traj(r,v,params):
    if isinstance(r,list):
        r=np.array(r)
    if isinstance(v,list):
        v=np.array(v)    
    n_new=params['n']
    n_old=len(r)
    vision=params['vision']
    avoid_factor=params['avoid_factor']
    match_factor=params['match_factor']
    L=params['L']
    margin=params['margin']
    turn_factor=params['turn_factor']
    centering_factor=params['centering_factor']
    max_speed=params['max_speed']
    min_speed=params['min_speed']
    v = v +  ( F_avoid(r,vision, avoid_factor) + F_align(r,v,vision,match_factor) + 
            F_cohesion(r,vision,centering_factor) + mind_edges(r,L,margin, turn_factor) )
    v = limit_speed(v,min_speed,max_speed)
    r,v = add_more_birds(r,v,n_old,n_new, params)
    r = r+v
    return r,v


def add_more_birds(r,v,n_old,n_new, params):
    L = params['L']
    margin = params['margin']
    min_speed = params['min_speed']
    max_speed = params['max_speed']
    if not(n_new is None):
        if n_old>n_new:
            if n_new>0:
                r=r[:n_new,:]
                v=v[:n_new,:]
        if n_old<n_new:
            n_diff=n_new-n_old
            r_diff, r_diff = initialize_birds(n_diff, L, margin, min_speed, max_speed)
            v_diff, v_diff = initialize_birds(n_diff, L, margin, min_speed, max_speed)
            r = np.vstack((r,r_diff))
            v = np.vstack((v,v_diff))
    return r,v



n_default = 100
avoid_factor_default = 3
match_factor_default = 3
centering_factor_default = 3
L_default = 200
vision_default =20
margin_default  = 10
turn_factor_default  = 1.5
max_speed_default = 5
min_speed_default = 2
frame_duration_default = 70
marker_size =3
marker_symbol ='circle' #arrow, square, diamond, star, cross, 
color_scheme = 'sunset'
#nice colors:  mint, plasma, rainbow, sunset, sunsetdark, viridis


params = {  
    'n':n_default,
    'vision':vision_default,
    'avoid_factor':avoid_factor_default,
    'match_factor':match_factor_default,
    'L':L_default,
    'margin':margin_default,
    'turn_factor':turn_factor_default,
    'centering_factor':centering_factor_default,
    'frame_duration': frame_duration_default,
    'max_speed':max_speed_default,
    'min_speed':min_speed_default }

defaults = params.copy()



animation_running = False
pause_event = Event()
r,v = initialize_birds(n_default, L_default, margin_default, min_speed_default, max_speed_default)
x_data=r[:,0]
y_data=r[:,1]
z_data=r[:,2]








    # Layout
app.layout = html.Div(
    [
        html.Div(
            [
                dcc.Graph(
                    id='scatter-plot',
                    style={'width': '100%', 'height': '100vh'}, 
                    config={'displayModeBar': False} 
                )                
            ],
            style={'flex': '0 0 60%', 'padding': '10px', 'box-sizing': 'border-box'}  
        ),        
        html.Div(
            [
                html.Div(
                    [   
                        html.Div(
                            children=[
                    html.H3("A Flock of Birds  (Boids Algorithm)"),
                    html.P("Boids algorithm simulates the movement of a flock of birds by having each bird care only about three simple adjustable parameters:"),
                    html.Ul([
                        html.Li(["Birds want to ", html.Em("avoid", style={'textDecoration' : 'underline', 'color' : 'darkgreen'}) , " collision and steer away from other birds."]),
                        html.Li(["Birds want to ",  html.Em("align" ,style={'textDecoration' : 'underline', 'color' : 'darkgreen'}), " their flight direction and speed with their flockmates."]),
                        html.Li(["Birds want to head towards the ", html.Em("center", style={'textDecoration' : 'underline', 'color' : 'darkgreen'}), " of the flock."])
                    ]),
                    html.P(["The birds make their decisions based on what (how far) they can see, which you can control by the ", html.Em("Field of Vision", style= {'textDecoration' : 'underline', 'color' : 'darkgreen'}), " slider. Also, as in real life, birds have minimum and maximum flying ", html.Em("speed", style= {'textDecoration' : 'underline', 'color' : 'darkgreen'}), "."]),
                    html.P("Programatically, Boids algorithm is usually implemented using object oriented programming, looping over bird as objects. As an experiment, I implemented the algorithm using entirely vectorization. I used neither objects nor for-loops."),
                    html.P(html.Em("If the animation is lagging, decrease the number of birds or increase the frames duration.", style={'fontSize':12}))
                ], 
                style={
                                'fontSize': 14, 
                                'color': 'black',  
                                'textAlign': 'justify',  
                                'paddingRight': '10px',
                                'margin-bottom': '20px'
                            }
                        ),
                        html.Div(
    [
        html.Button('Play', id='play-button', n_clicks=0, style={'margin-right': '10px', 'display': 'inline-block', 'width': '80px', 'height': '30px', 'background-color':'rgb(237,245,255)','border-color':'rgb(214,220,230)'}),
        html.Button('Pause', id='pause-button', n_clicks=0, style={'margin-right': '10px', 'display': 'inline-block', 'width': '80px', 'height': '30px', 'background-color':'rgb(237,245,255)','border-color':'rgb(214,220,230)'}),
        html.Button('Reset', id='reset-button', n_clicks=0, style={'margin-right': '10px', 'display': 'inline-block', 'width': '80px', 'height': '30px', 'background-color':'rgb(237,245,255)','border-color':'rgb(214,220,230)'}),    
        # Container for Number of Birds
        html.Div([
            html.Label('Number of Birds', style={'display': 'block', 'margin-bottom': '5px'}),
            dcc.Input(
                id='n_birds',
                type='number',
                placeholder='',
                value=n_default,
                min=1,
                debounce=False,
                required=True,
                style={'display': 'inline-block', 'width': '85px', 'height': '25px', 'vertical-align': 'bottom'}
            )
        ], style={'display': 'inline-block', 'margin-right': '20px', 'text-align': 'left', 'vertical-align': 'bottom'}),
        # Container for Frame Duration
        html.Div([
            html.Label('Frame duration (ms)', style={'display': 'block', 'margin-bottom': '5px'}),
            dcc.Input(
                id='frame_duration',
                type='number',
                placeholder='',
                value=frame_duration_default,
                min=1,
                debounce=True,
                required=True,
                style={'display': 'inline-block', 'width': '80px', 'height': '25px', 'vertical-align': 'bottom'}
            )
        ], style={'display': 'inline-block', 'text-align': 'left', 'vertical-align': 'bottom'})
    ],
    style={'text-align': 'left', 'padding': '10px 0', 'display': 'flex', 'align-items': 'flex-end'}
),
                        html.Div([html.Label('Avoid Factor', style={'display': 'block', 'margin-bottom': '5px'}), dcc.Slider(0, 10, 0.01, value = avoid_factor_default, id='avoid_slider', tooltip={"placement": "right", "always_visible": True}, marks=None) ]),
                        html.Div([html.Label('Match Factor', style={'display': 'block', 'margin-bottom': '5px'} ), dcc.Slider(0, 10, 0.01, value=match_factor_default , id='match_slider', tooltip={"placement": "right", "always_visible": True}, marks=None)], style={'margin': '0 0'}),
                        html.Div([html.Label('Centering Factor', style={'display': 'block', 'margin-bottom': '5px'}), dcc.Slider(0, 10, 0.01, value=centering_factor_default , id='centering_slider', tooltip={"placement": "right", "always_visible": True}, marks=None)], style={'margin': '0 0'}),
                        html.Div([html.Label('Field of Vision', style={'display': 'block', 'margin-bottom': '5px'}), dcc.Slider(0, 350, 0.1, value=vision_default , id='vision_slider', tooltip={"placement": "right", "always_visible": True}, marks=None)], style={'margin': '0 0'}),
                        html.Div([html.Label('Turn Factor',style={'display': 'block', 'margin-bottom': '5px'}), dcc.Slider(0, 10, 0.01, value=turn_factor_default , id='turn_slider', tooltip={"placement": "right", "always_visible": True}, marks=None)], style={'display':'none'}),
                        html.Div([html.Label('Speed', style={'display': 'block', 'margin-bottom': '5px'}), dcc.RangeSlider(0, 10, 0.01, value=[min_speed_default , max_speed_default], id='speed_slider', tooltip={"placement": "bottom", "always_visible": True}, marks=None)], style={'margin': '0 0'})
                    ],
                    style={'margin-top': 'auto'}  # Align sliders at the bottom
                )
            ],
            style={'flex': '0 0 34%', 'padding': '0px', 'box-sizing': 'border-box', 'display': 'flex', 'flex-direction': 'column'}  # Right takes 34% of width
        ),
        # Interval and Store components for real-time updates
        dcc.Interval(id='interval-component', interval=frame_duration_default, n_intervals=0, disabled=True),  # Controls the real-time updates
        dcc.Store(id='is_playing', data=False),
        dcc.Store(id='r', data=r.tolist()),
        dcc.Store(id='v', data=v.tolist()),
        dcc.Store(id='params', data=params),
        dcc.Store(id='default_values', data=defaults),
    ],
    style={'display': 'flex', 'flex-direction': 'row', 'width': '100vw', 'height': '95vh', 'overflow': 'hidden',   
                                'fontFamily' : 'Arial', 'fontSize':13, 'color':'#333333'}  # Ensures full screen utilization
)






@app.callback(
    Output('interval-component', 'interval'),
    Input('frame_duration', 'value'),
    prevent_initial_call=True
)
def change_frame_interval_duration(frame_duration):
    # Change the interval value based on the number of clicks
    if frame_duration is None or frame_duration<1:
        new_interval = frame_duration_default
    else: 
        new_interval = frame_duration 
    return new_interval



def draw_graph(r,params):
    x_data=r[:,0]
    y_data=r[:,1]
    z_data=r[:,2]
    L=params['L']

    # Create the scatter plot
    fig = go.Figure(
        data=[go.Scatter3d(
            x=x_data, y=y_data, z=z_data, mode='markers',
            marker=dict(color=np.linspace(0, 1, params['n']), colorscale=color_scheme, size=marker_size, symbol=marker_symbol)
            )],
        layout= dict(
            autosize=True,
            scene = dict(
                xaxis = dict(
                    range=[-L,L],
                    autorange=False
                ),
                yaxis = dict(
                    range=[-L,L],
                    autorange=False
                ),
                zaxis = dict(
                    range=[-L,L],
                    autorange=False
                ),
                aspectmode = 'cube',
                camera= {
                    'center': { 'x': 0, 'y': 0, 'z': 0 }, 
                    'eye': { 'x': -1.7, 'y': -1.7, 'z': -1.7 },   #   cool --> 'eye': { 'x': -1.7, 'y': -1.7, 'z': -1.7 },   
                    'up': { 'x': 0, 'y': 0, 'z': 1.5 }
                }
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
    )
    return fig



def initialize_graph(params):
    n = params['n']
    L = params['L']
    margin=params['margin']
    min_speed = params['min_speed'] 
    max_speed = params['max_speed']
    r,v = initialize_birds(n,L,margin,min_speed,max_speed)
    fig = draw_graph(r,params)
    return r,v,fig





# Update the plot in real-time
@app.callback(
    Output('scatter-plot', 'figure',allow_duplicate = True),
    Output('r','data',allow_duplicate = True),
    Output('v','data',allow_duplicate = True),
    Input('interval-component', 'n_intervals'),
    Input('params','data'),
    State('r','data'),
    State('v','data'))
def update_graph(n_intervals,params,r,v):
    # this is automatically updated everytime the interval-component is updated, thus providing the animation.
    r,v = update_traj(r, v,params)  
    fig = draw_graph(r,params)
    return fig,r,v



@app.callback(
    Output('params','data',allow_duplicate = True),
    Input('n_birds','value'), 
    Input('vision_slider','value'),
    Input('avoid_slider','value'), 
    Input('match_slider', 'value'),
    Input('centering_slider','value'), 
    Input('turn_slider', 'value'),
    Input('speed_slider','value'), 
    State('params','data')
)
def update_params(n_birds,vision_slider,avoid_slider,match_slider,centering_slider,turn_slider, speed_slider,params):
    params['n']=n_birds
    params['vision']=vision_slider
    params['avoid_factor']=avoid_slider
    params['match_factor']=match_slider
    params['centering_factor']=centering_slider
    params['turn_factor']=turn_slider
    params['min_speed']=min(speed_slider)
    params['max_speed']=max(speed_slider)
    return params


# Play button callback to start the animation
@app.callback(
    Output('interval-component', 'disabled', allow_duplicate = True),
    [Input('play-button', 'n_clicks'),
    Input('pause-button', 'n_clicks')])
def control_animation(play_clicks, pause_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return True
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'play-button':
        animation_running = True
        pause_event.clear()
        return False  # Enable the interval to start real-time updates
    elif button_id == 'pause-button':
        animation_running = False
        pause_event.set()
        return True  # Disable the interval to stop real-time updates
    return True


@app.callback(
    Output('params','data',allow_duplicate = True),
    Output('n_birds','value'), 
    Output('vision_slider','value'),
    Output('avoid_slider','value'), 
    Output('match_slider', 'value'),
    Output('centering_slider','value'), 
    Output('turn_slider', 'value'),
    Output('speed_slider','value'),
    Output('interval-component', 'disabled',allow_duplicate = True),
    Output('frame_duration','value'),
    Output('scatter-plot', 'figure',allow_duplicate = True),
    Output('r','data',allow_duplicate = True),
    Output('v','data',allow_duplicate = True),
    Input('reset-button', 'n_clicks'),
    State('default_values','data')
)
def reset_app(reset,defaults):
    params = defaults
    n_birds = n_default 
    vision_slider = vision_default
    avoid_slider = avoid_factor_default
    match_slider = match_factor_default
    centering_slider = centering_factor_default
    turn_slider = turn_factor_default
    speed_slider = [min_speed_default, max_speed_default]
    frame_duration = frame_duration_default
    interval_disabled = True
    r,v, fig = initialize_graph(params)
    return params,n_birds,vision_slider,avoid_slider,match_slider,centering_slider,turn_slider,speed_slider,interval_disabled,frame_duration,fig,r,v  


# Run the app
if __name__ == "__main__":
    app.run_server(host='0.0.0.0', port=int(os.environ.get('PORT', 8050)))

    
