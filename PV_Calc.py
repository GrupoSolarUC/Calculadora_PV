"""
Herramienta computacional para el pre-diseño y evaluación de sistemas solares 
fotovoltaicos a nivel domiciliario en Python, en base horaria

Pre-diseño geométrico y eléctrico (disposición módulos serie/paralelo, 
selección de inversor) y geométrica (filas, módulos por fila) para un área 
determinada.

Modelación de sombreado propio.

Indicadores económicos de desempeño: CAPEX, VAN, ROI.
"""
#%%
"""
Create the database for the inverters fromm SAM.
Requires installing PVLib
CSV file already created
"""
# import pandas as pd
# from pvlib import pvsystem
# data=pvsystem.retrieve_sam('CECInverter')
# data=data.drop(data.index[[14,15]])
# data.to_csv('CECInverter_SAM.csv', index=False)

#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from datetime import datetime, date
from bisect import bisect_right

input_data = {
  "Pstc": 320, #Nominal power at STC [W]
  "Tnoct": 42, #Normal Operating Cell Temperature [deg]
  "beta_P": -0.41/100, #Power thermal de-rate [1/K]
  "beta_V": -0.33/100, #Voltage thermal de-rate [1/K]
  "Voc": 46, #Module open-circuit voltage [V]
  "Vmp": 37.4, #Module voltage at maximum power point at STC [V]
  "MODULE_L": 1.7, #Module length [m]
  "MODULE_W": 1, #Module width [m]
  "module_cost": 180000/340, #Cost of one module [CLP/Wp] 
  "inverter_cost": 430000/3000, #Cost of inverter [CLP/Wac]
  "Nyears": 20, #Proyect Lifetimme [years]
  "degadation": 0.5/100, #Yearly degradation rate
  "inflation": 3/100, #Expected inflation [%]
  
  "D": [76, 90, 84, 91, 103, 162, 192, 93, 104, 97, 85, 80], # Demanda para cada mes [kWh]
  "K1": 76, #Consumo del ultimo mes [kWh]
  "K2": 76*130, #Consumo del ultimo mes [Clp]
  "MAXMODULES": 12, #Cuántos módulos instalará
  "AMAX": 10, #Área máxima disponible para el sistema [m²]
  "latitude": -33.4976, #deg
  "longitude": -70.6065, #deg
  "elevation": 576, #[m]
  "UTC": -4,
  "ALBEDO": 0.2 #[-]
}

# Electricity_cost
electricity_cost=input_data["K2"]/input_data["K1"]

# Define module
Pstc=input_data["Pstc"] #Nominal power at STC [W]
Tnoct=input_data["Tnoct"] #Normal Operating Cell Temperature [deg]
beta_P=input_data["beta_P"] #Power thermal de-rate [1/K]
beta_V=input_data["beta_V"] #Voltage thermal de-rate [1/K]
Voc=input_data["Voc"] #Module open-circuit voltage [V]
Vmp=input_data["Vmp"] #Module voltage at maximum power point at STC [V]
MODULE_L=input_data["MODULE_L"] #Module length [m]
MODULE_W=input_data["MODULE_W"] #Module width [m]
Area_module=MODULE_L*MODULE_W #Module area [[m²]]
module_cost=input_data["module_cost"] #Cost of one module [CLP/Wp] 
inverter_cost=input_data["inverter_cost"] #Cost of inverter [CLP/Wac]
Nyears=input_data["Nyears"] #Proyect Lifetimme [years]
degadation=input_data["degadation"] #Yearly degradation rate
inflation=input_data["inflation"] #Expected inflation [%]

# Hourly demand #
# demand_profile - Function definition
def demand_profile():
    """
    #Create the hourly demand profile for the year    
   
    #No external INPUTS 
    
    #OUTPUT
    DEMANDPROFILE
    """
    M=np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    H=np.concatenate((np.array([0]),24*M.cumsum()))-1   
   # monthstrings=["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]  
    
    #Monthly demand for year
    # D=np.zeros((12,1))
    # for i in range(0,12,1):
    #     # D[i]=float(input("Introduzca su demanda (kWh) para el mes de " + monthstrings[i] + ": "))#############################################################################################################
    #     D[i]=100
    D=np.array(input_data["D"])    
    
    #FOR WEEKDAYS
    FRIDGE=np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    COOKING=np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0])
    WASHING=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,])
    HWATER=np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,])
    LIGHT=np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,])
    APPL=np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,])
    
    DEMAND=(1000*10/60)*FRIDGE+2000*COOKING+1000*WASHING+1000*HWATER+100*LIGHT+250*APPL;
    WEEKDAYDEMAND=DEMAND
    
    #FOR WEEKENDDAYS
    FRIDGE=np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0,])
    COOKING=np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,])
    WASHING=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0,])
    HWATER=np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,])
    LIGHT=np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,])
    APPL=np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,])
    
    DEMAND=(1000*10/60)*FRIDGE+2000*COOKING+1000*WASHING+1000*HWATER+100*LIGHT+250*APPL;
    WEEKENDDAYDEMAND=DEMAND
   
    MONTH=1
    DAY=0
    DAYTYPE=0
    WEEKDAYTYPE=0
    DEMANDPROFILE=np.zeros(8760)
    MONTHCOUNTER=[]
    for i in range(1,366,1):
        DAY=DAY+1;
        if DAY>M[MONTH-1]:
            DAY=1
            MONTH=MONTH+1
        DAYTYPE=DAYTYPE+1
        if DAYTYPE>5 and WEEKDAYTYPE==0:
            WEEKDAYTYPE=1
        elif DAYTYPE>7:
            WEEKDAYTYPE=0
            DAYTYPE=1
        if WEEKDAYTYPE==0:
            #WEEKDAY
            DEMANDPROFILE[(i-1)*24:(i*24)]=WEEKDAYDEMAND
        else: 
            #WEEKEND
            DEMANDPROFILE[(i-1)*24:(i*24)]=WEEKENDDAYDEMAND
        MONTHCOUNTER=np.concatenate((MONTHCOUNTER,MONTH*np.ones((24),float)))        
    DEMANDPROFILE.shape=(1,int(DEMANDPROFILE.size))
    for i in range(0,12,1):
        dummy=DEMANDPROFILE[0,H[i]+1:H[i+1]+1]
        DEMANDPROFILE[0,H[i]+1:H[i+1]+1]=D[i]*dummy/dummy.sum()
    return DEMANDPROFILE       
# demand_profile - End of definition

demand=demand_profile()
demand=demand*1000

maxdemand=np.minimum(8000,input_data["MAXMODULES"]*Pstc)

#Query the user as to what is the maximum area available
#AMAX=float(input("Cuál es el área máxima (m^2) disponible para el sistema: "))#############################################################################################################
AMAX=input_data["AMAX"]

#Location information
latitude=input_data["latitude"]
longitude=input_data["longitude"]
elevation=input_data["elevation"]
UTC=input_data["UTC"]

#Read meteorological database 
DDATOS = pd.read_csv('Santiago.csv', header=None, sep=';')
DDATOS = np.roll(DDATOS, 0, axis=0)

YEAR=DDATOS[:,0]
MONTH=DDATOS[:,1]
DAY=DDATOS[:,2]
HOUR=DDATOS[:,3]
GHI=DDATOS[:,5]
DNI=DDATOS[:,6]
DHI=DDATOS[:,7]
TDRY=DDATOS[:,8]
TWET=DDATOS[:,9]
RH=DDATOS[:,10]
WS=DDATOS[:,12]
WD=DDATOS[:,13]

# date_correction - Function definition
def date_correction(y,m,d,hh,mm,ss):
    monthdays=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
    # now to test if it's a leap year
    ly=y
    while ly>0:
        ly=ly-4
    if ly==0:
        #it's a leap year, add 1 day to february
        monthdays[1]=monthdays[1]+1

    #seconds correction
    if ss>=60: 
        while ss>=60: 
            mm=mm+1
            ss=ss-60
    elif ss<0:
        while ss<0:
            mm=mm-1
            ss=60+ss

    #minutes correction
    if mm>=60: 
        while mm>=60:
            hh=hh+1
            mm=mm-60
    elif mm<0: 
        while mm<0:
            hh=hh-1
            mm=60+mm    
    #hours correction
    if hh>=24: 
        hh=hh-24
        d=d+1
    elif hh<0:
        hh=hh+24
        d=d-1  
    
    #month and year correction 
    if m>12 :
        m=m-12
        y=y+1  
    elif m<1:
        m=m+12
        y=y-1     

    #days correction
    if d>monthdays[int(m-1)]: 
        d=d-monthdays[int(m-1)]
        m=m+1
    elif d<1:
        if m>0 :
            d=monthdays[int(m-2)]-d
            m=m-1
        else:
            d=monthdays[11]-d
            m=12
            y=y-1
    
    #recheck month & year
    if m>12: 
        m=m-12
        y=y+1  
    elif m<1:
        m=m+12;
        y=y-1;
    
    return [y,m,d,hh,mm,ss]
# date_correction - End of definition

for i in range(0,GHI.size-1,1): 
    dummy=np.array(date_correction(YEAR[i],MONTH[i],DAY[i],HOUR[i],0,0))
    YEAR[i]=dummy[0]
    MONTH[i]=dummy[1]
    DAY[i]=dummy[2]
    HOUR[i]=dummy[3]
del dummy

# solpos2 - Function definition
def solpos2(YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,TZ,latitude,longitude):
    """
    #Solar position    
   
    #INPUTS: 
    YEAR 
    MONTH 
    DAY 
    HOUR 
    LATITUDE 
    LONGITUDE
    
    #OUTPUTS
    Ha Hour Angle
    El Solar Elevation Angle
    Az Solar Azimuth Angle
    Dec Declination Angle
     
    """
    daynumber = date(int(YEAR), int(MONTH), int(DAY)).timetuple().tm_yday
    daynumber=daynumber+((HOUR-TZ)/24)+(MINUTE/(24*60));
    if YEAR%4==0:
         B=math.radians((daynumber-1)*(360/366))
    else:
         B=math.radians((daynumber-1)*(360/365))
     
    Dec=(180/np.pi)*(0.006918-(0.399912*np.cos(B))+(0.070257*np.sin(B))-(0.006758*np.cos(2*B))+(0.000907*np.sin(2*B))-(0.002697*np.cos(3*B))+(0.00148*np.sin(3*B)))
    EOT=229.2*(0.000075+(0.001868*np.cos(B))-(0.032077*np.sin(B))-(0.014615*np.cos(2*B))-(0.04089*np.sin(2*B)))
     
     #AST and Ha is calculated as the excel sheet file 
    AST=(((HOUR+MINUTE/60+SECOND/3600))/24)*1440+(EOT)+4*longitude-60*TZ;
    if AST>1440:
        AST=AST-1440
     
     #hour angle
    Ha=180+AST/4
    if AST/4>=0:
        Ha=AST/4-180
     
     #Elevation angle (Duffie)
    El = (180/np.pi)*np.arcsin(np.sin(math.radians(Dec))*np.sin(math.radians(latitude))+np.cos(math.radians(Dec))*np.cos(math.radians(latitude))*np.cos(math.radians(Ha)))
     
     #Azimuth Angle (Duffie)
     #Az = np.arcsin(-np.cosd(math.radians(Dec)).*np.sin(math.radians(Ha))/np.cos(math.radians(El)))
    Az=np.sign(Ha)*np.abs( (180/np.pi)*np.arccos(((np.sin(math.radians(El)) * np.sin(math.radians(latitude)))-np.sin(math.radians(Dec))) / (np.cos(math.radians(El))*np.cos(math.radians(latitude)))) )
    return [Ha, El, Az, Dec, AST] # [Deg, Deg, Deg, Deg, Deg]
# solpos2 - End of definition


hourangle=np.zeros((YEAR.size,1)) #Deg
elevationangle=np.zeros((YEAR.size,1)) #Deg
azimuthangle=np.zeros((YEAR.size,1)) #Deg
declinationangle=np.zeros((YEAR.size,1)) #Deg
ast=np.zeros((YEAR.size,1)) #Deg
daynumber=np.zeros((YEAR.size,1))
epsilon=np.zeros((YEAR.size,1))
Io=np.zeros((YEAR.size,1))
for i in range(0,YEAR.size,1):
    dummy=solpos2(YEAR[i],MONTH[i],DAY[i],HOUR[i],0,0,UTC+0.5*0,latitude,longitude)
    hourangle[i]=dummy[0]
    elevationangle[i]=dummy[1]
    azimuthangle[i]=dummy[2]
    declinationangle[i]=dummy[3]
    ast[i]=dummy[4]
    daynumber[i] = date(int(YEAR[i]), int(MONTH[i]), int(DAY[i])).timetuple().tm_yday
    epsilon[i]=1+0.0334*np.cos(0.01721*daynumber[i]-0.0552)
    Io[i]=np.maximum(0,1367*epsilon[i]*np.sin(math.radians(elevationangle[i])))   
del dummy
   
#Default values: plane tilted at latitude, oriented towards the Sun
SLOPE=np.abs(latitude)
if latitude>=0:
    ORIENT=0
else:
    ORIENT=180

# POA_HDKR_FP - Function definition
def POA_HDKR_FP(GHI,DNI,DHI,io,sza,saa,SLOPE,ORIENT):
    """
    Calculates the irradiance on a fixed tilted plane, using the HDKR 
    irradiance transposition model.
    
    #Inputs
    GHI: Global Horizontal Irradiance (W/m²)
    DNI: Direct Normal Irradiance (W/m²)
    DHI: Diffuse Horizontal Irradiance (W/m²)
    io: Extraterrestrial horizontal irradiance (W/m²)
    sza: Sun zenith angle (deg)
    saa: Sun azimuth angle (deg)
    SLOPE: Plane of array inclination (deg)
    ORIENT: Plane of array orientation (deg)
        
    #Outputs
    IT: Tilted plane irradiance (W/m²)
    ITDIRECT: Portion of IT from Direct and circumsolar irradiance (W/m²)
    ITDIFFUSE: Portion of IT from Diffuse irradiance (W/m²)
    ITGLOBAL: Portion of IT from ground reflected irradiance (W/m²)
    """

    sa=GHI.size
    
    #prefiltering of variables and calculation of HDKR parameters
    if DNI<0:
        DNI=0
    if io<=0:
        Ai=0
    else:
        Ai=DNI/io
    Ai=np.minimum(1,np.maximum(0,Ai))
    
    if io<=0 or GHI<=0:
        f=0
    else:
        f=np.real( math.sqrt((DNI*np.cos(math.radians(sza)))/GHI) ) 
    f=np.maximum(0,f)
    
    #incidence angle for inclined plane pointing to the sun and slope=latitude
    ORIENT=np.ones([sa])*ORIENT
    if SLOPE<0:
        SLOPE=np.ones([sa])*np.abs(SLOPE)
        ORIENT=ORIENT+180
    else:
        SLOPE=np.ones([sa])*SLOPE
    inc_ang_fix=(180/np.pi)*np.arccos(np.minimum(1,np.maximum(-1,(((np.cos(math.radians(sza))*np.cos(math.radians(SLOPE)))+(np.sin(math.radians(sza))*np.sin(math.radians(SLOPE))*np.cos(math.radians(saa-ORIENT))))))))

    #HDKR Model proper
    ITDIRECT=((DNI*np.cos(math.radians(inc_ang_fix)))+(DHI*Ai*np.cos(math.radians(inc_ang_fix))/np.cos(math.radians(sza))))
    if inc_ang_fix>=90:
        ITDIRECT=0
    if sza>=89:
        ITDIRECT=0
    ITDIFFUSE=DHI*(1-Ai)*(1+np.cos(math.radians(SLOPE)))*0.5*(1+f*((np.sin(math.radians(SLOPE/2)))**3))
    if inc_ang_fix>=90:
        ITDIFFUSE=DHI*(1+np.cos(math.radians(SLOPE)))*0.5*(1+f*((np.sin(math.radians(SLOPE/2)))**3))
    ITGLOBAL=GHI*input_data["ALBEDO"]*(1-np.cos(math.radians(SLOPE)))*0.5
    if inc_ang_fix>=90:
        ITDIRECT=0
    if io<=0:
        ITDIRECT=0
    if io<=0:
        ITDIFFUSE=0
    if io<=0:
        ITGLOBAL=0
    IT=ITDIRECT+ITDIFFUSE+ITGLOBAL
    return [IT, ITDIRECT, ITDIFFUSE, ITGLOBAL, inc_ang_fix]
# POA_HDKR_FP - End of definition

#Calculate POA irradiance
IT=np.zeros((YEAR.size,1)) 
ITDIRECT=np.zeros((YEAR.size,1)) 
ITDIFFUSE=np.zeros((YEAR.size,1)) 
ITGLOBAL=np.zeros((YEAR.size,1)) 
TM=np.zeros((YEAR.size,1))
TC=np.zeros((YEAR.size,1))
P=np.zeros((YEAR.size,1))
Voct=np.zeros((YEAR.size,1))
Vmpt=np.zeros((YEAR.size,1))
inc_ang_fix=np.zeros((YEAR.size,1)) 
for i in range(0,YEAR.size,1):    
    dummy=POA_HDKR_FP(GHI[i],DNI[i],DHI[i],Io[i],90-elevationangle[i],azimuthangle[i],SLOPE,ORIENT) #Perez HDKR
    IT[i]=dummy[0]
    ITDIRECT[i]=dummy[1]
    ITDIFFUSE[i]=dummy[2]
    ITGLOBAL[i]=dummy[3]
    inc_ang_fix[i]=dummy[4]
    
    #Estimate temperature with simple correlation
    #Module
    TM[i]=max(TDRY[i],TDRY[i]+(IT[i]*math.exp(-3.56-0.075*WS[i])))
    #Cell
    TC[i]=TM[i]+3*IT[i]/1000
    
    #Estimate total annual production for just one module
    P[i]=Pstc*(IT[i]/1000)*(1+beta_P*(TC[i]-25))
    
    #Let's assume that the modules operate somewhere in between Vmp<V<Voc
    Voct[i]=Voc*(1+beta_V*(TC[i]-25)) #Voc, corrected by temperature (V)
    Vmpt[i]=Vmp*(1+beta_V*(TC[i]-25)) #Voc, corrected by temperature (V)
Vlim=np.array([min(Vmpt),max(Voct)])
del dummy
 

#Define inverter
data = pd.read_csv('CECInverter_SAM.csv')
inverter_names=list(data.columns.values)
data=data.to_numpy()

#get total number of inverters in the database
inverterID=np.linspace(1,int(data.shape[1]),int(data.shape[1]))


#With the maximum demand, filter inverter list to retain all inverters
#with maximum Pmaxdemand<Pac<1.1*maxdemand
dummy=data[2,:].astype(np.float64)

#check that the maxdemand it's at least equal to the smallest inverter  
if min(dummy)>maxdemand:
    maxdemand=min(dummy)+1e-6
    
inverterfilter=np.ones([dummy.size])*((dummy>maxdemand)*(dummy<(1.1*maxdemand)))

#Then, filter inverters so that Vlim(1)>Vmin_in 
dummy=data[12,:].astype(np.float64)
inverterfilter=inverterfilter*((dummy*(inverterfilter==1))>Vlim[0])

#Then, select the inverter with the highest efficiency, for this example,
dummy=data[1,:].astype(np.float64)/data[3,:].astype(np.float64)
inverterID=inverterID*(inverterfilter==1)

inverterfilter=(dummy*(inverterfilter==1))==max(dummy*(inverterfilter==1))
inverterID=inverterfilter.ravel().nonzero()
inverterID=int(inverterID[0][0]) #This is the inverter to use!

#Then, get inverter voltage limits (V)
Vmin=data[12,inverterID]
Vmax=data[13,inverterID]

#Then, let's get the inverter max input power (W)
Pmax_ac=data[2,inverterID]
Pmax_dc=data[3,inverterID]

#Limits for amount of modules in series
#N_series_max=np.maximum(1,np.floor(np.linalg.lstsq(Vlim,np.array([Vmin,Vmax]),rcond=None)[0]))
N_series_max=np.floor(Vmax/max(Voct))
N_series_min=np.ceil(Vmin/min(Vmpt))
N_series=np.floor(np.median(np.array([N_series_max, N_series_min]))) 

#Final Array evaluation, based on the best config
N_parallel=np.floor(input_data["MAXMODULES"]/N_series)
if N_parallel<1:
    N_parallel=1
    input_data["MAXMODULES"]=N_series ##################################################### ARMANDO #################################################
    print('Se requiere de al menos ', N_series, ' módulos fotovoltaicos')    

#Final GCR for shading
GCR=Area_module*N_series*N_parallel/AMAX
ROW_SPACING=MODULE_L/GCR
AREA_L=np.sqrt(AMAX) #Assumes a square terrain
N_ALONG_BOTTOM=np.floor(AREA_L/MODULE_W) #modules per geometrical row
N_ROWS=np.ceil(N_series*N_parallel/N_ALONG_BOTTOM) #geometrical rows

Py=MODULE_L*np.cos((np.pi/180)*SLOPE)+(np.cos((np.pi/180)*azimuthangle)*np.sin((np.pi/180)*SLOPE)/np.tan((np.pi/180)*elevationangle))
Px=MODULE_L*(np.sin((np.pi/180)*SLOPE)*np.sin((np.pi/180)*azimuthangle)/np.tan((np.pi/180)*elevationangle))
g=np.abs(ROW_SPACING*Px/Py)
g=g*(Py!=0).astype(np.float32)
g=(g>MODULE_L).astype(np.float32)*MODULE_L + (g<=MODULE_L).astype(np.float32)*g

#SHADOW HEIGHT
HS=MODULE_L*(1-ROW_SPACING/Py)
HS=HS*(Py!=0).astype(np.float32)
HS=(g<0).astype(np.float32)*np.abs((g<0).astype(np.float32)*HS) + (g>=0).astype(np.float32)*HS
HS=(HS>MODULE_L).astype(np.float32)*MODULE_L + (HS<=MODULE_L).astype(np.float32)*HS
SHADING_SAM=np.maximum(0,np.minimum(1,HS*(MODULE_W-g)/Area_module))
SHADING_SAM=(elevationangle<=0).astype(np.float32)*1 + (elevationangle>0).astype(np.float32)*SHADING_SAM

#FOR FIRST ROW THERE IS NO SHADING
P_FIRST_ROW=P*N_ALONG_BOTTOM 

#FOR THE REMAINDER OF ROWS, WE APPLY SHADING
P_N_ROWS=P*(1-SHADING_SAM)*(N_ROWS-1)*N_ALONG_BOTTOM


#Construct system and simulate actual production
#That would come here

#Calculate energy flows
Pdc_raw=P_FIRST_ROW+P_N_ROWS #Raw DC power [W]
P_shading=(P*N_parallel*N_series)-Pdc_raw #Power lost to shading [W]
P_clipped=np.maximum(0,Pdc_raw-Pmax_dc) #Power lost to clipping (W)
Pac_net=np.minimum(demand.transpose(),(Pdc_raw-P_clipped)*(Pmax_ac/Pmax_dc)) #Net AC power (W)
P_loss=(Pdc_raw-P_clipped)*(1-Pmax_ac/Pmax_dc) #Power loss to inefficiency (W)
P_sell=Pdc_raw-P_clipped-P_loss-Pac_net; #Power that could be sold to grid (W)
P_sell=P_sell*(P_sell>0).astype(int) #Remove values less than 0

#Calculate economic indicators
CAPEX_MODULE=module_cost*N_series*N_parallel*Pstc
CAPEX_INVERTER=inverter_cost*Pmax_ac
CAPEX=(CAPEX_MODULE+CAPEX_INVERTER)*1.02 #a 2% increase of overhead is assumed


#What was being originally paid for energy consumption
ORIGINAL_ENERGY_EXPENSE=(demand.sum()/1000)*electricity_cost

#What will be generated using a net-billing scheme
ENERGY_REVENUE=(P_sell.sum()/1000)*electricity_cost*0.5

#What will be paid for energy consumption, using the system
ENERGY_EXPENSE=(sum(np.transpose(demand)-Pac_net)/1000)*electricity_cost

#Let's evaluate the system along its lifecycle
Energy_Bought=np.zeros((Nyears,1))
Energy_Sold=np.zeros((Nyears,1))
Inflation_Factor=np.ones((Nyears,1))
for i in range(1,Nyears+1,1): 
    Pdc_raw_Y=Pdc_raw*(0.5*((1-degadation)**(i)+(1-degadation)**(i-1)))
    P_clipped_Y=np.maximum(0,Pdc_raw_Y-Pmax_dc)
    Pac_net_Y=np.minimum(np.transpose(demand),(Pdc_raw_Y-P_clipped_Y)*(Pmax_ac/Pmax_dc))
    P_loss_Y=(Pdc_raw_Y-P_clipped_Y)*(1-Pmax_ac/Pmax_dc)
    P_sell_Y=Pdc_raw_Y-P_clipped_Y-P_loss_Y-Pac_net_Y
    Energy_Bought[i-1,0]=sum(np.transpose(demand)-Pac_net_Y)
    Energy_Sold[i-1,0]=sum(P_sell_Y)
    Inflation_Factor[i-1,0]=(1-inflation)**(i-1)

#Let's calculate the cash flows

#Business As Usual case, CASH_FLOWS without the PV system
Cash_Flow_BAU=-(demand.sum()/1000)*electricity_cost*Inflation_Factor

#System cash flow

#Initial Investment
Cash_Flow=-CAPEX*np.ones((Nyears,1))
Cash_Flow[1:Nyears+1,0]=0

#Bought energy when using the system
Cash_Flow=Cash_Flow-(Energy_Bought/1000)*Inflation_Factor*electricity_cost

#Sold energy, using a net-billing scheme
Cash_Flow=Cash_Flow+(Energy_Sold/1000)*Inflation_Factor*electricity_cost*0.5

#Project cash flow
NET_Cash_Flow=Cash_Flow-Cash_Flow_BAU

#Return of investment
ROI=NET_Cash_Flow.cumsum(axis=0)
VAN=float(ROI[int(ROI.size)-1])

### FIGURES ###
plt.rcParams["figure.figsize"] = [12, 8]
font = {'family' : 'DejaVu Sans',
        'size'   : 20}
plt.rc('font', **font)

#Figure(1)
plt.figure(1)
positions=range(len(ROI))
positions=range(1,21)
ROI.shape=(len(ROI),)
plt.bar(positions, ROI/1e+6, align='center', zorder=3)
plt.xlabel('Año [-]')
plt.ylabel('Flujo de caja acumulado [MMCLP]')
plt.grid(zorder=0)
plt.tight_layout(pad=2)
plt.savefig("Fig1", dpi=600, bbox_inches="tight")
plt.show()


#Figure (2)
plt.figure(2)
Pdc_raw_M=np.zeros(12)
P_clipped_M=np.zeros(12)
Pac_net_M=np.zeros(12)
P_loss_M=np.zeros(12)
P_shading_M=np.zeros(12)
P_sell_M=np.zeros(12)
IT_M=np.zeros(12)
MONTH.shape=(MONTH.size,1)
for i in range(0,12):   
    Pdc_raw_M[i]=sum(Pdc_raw*(MONTH==i+1))
    Pac_net_M[i]=sum(Pac_net*(MONTH==i+1))
    P_clipped_M[i]=sum(P_clipped*(MONTH==i+1))
    P_loss_M[i]=sum(P_loss*(MONTH==i+1))
    P_shading_M[i]=sum(P_shading*(MONTH==i+1))
    P_sell_M[i]=sum(P_sell*(MONTH==i+1))
    IT_M[i]=sum(IT*(MONTH==i+1))

positions=range(1,13)
plt.bar(positions, Pac_net_M/1000, color='r', zorder=3)
plt.bar(positions, P_sell_M/1000, bottom=Pac_net_M/1000, color='g', zorder=3)
plt.bar(positions, P_loss_M/1000, bottom=(Pac_net_M+P_sell_M)/1000, color='b', zorder=3)
plt.bar(positions, P_shading_M/1000, bottom=(Pac_net_M+P_sell_M+P_loss_M)/1000, color='m', zorder=3)
plt.bar(positions, P_clipped_M/1000, bottom=(Pac_net_M+P_sell_M+P_loss_M+P_shading_M)/1000, color='k', zorder=3)
plt.legend(['Suministrada','Vendida','Pérdidas','Sombras','Recorte'])
plt.xlabel('Mes [-]')
plt.ylabel('Energía [kWh]')
plt.grid(zorder=0)
plt.savefig("Fig2", dpi=600, bbox_inches="tight")
plt.show()

#Figure(3)
demand.shape=(1,demand.size)
monthdays=np.array([31,28,31,30,31,30,31,31,30,31,30,31])
monthhours=24*monthdays.cumsum()
MONTHSTRING=['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
plt.rcParams["figure.figsize"] = [12, 8]
fig, ax=plt.subplots(4,3, sharex=True, sharey=True)
fig.tight_layout(pad=2.0)

#Ene
i=0
hini=0
hend=monthhours[i]
ax[0,0].set_title(MONTHSTRING[i])
ax[0,0].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[0,0].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[0,0].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[0,0].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[0,0].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[0,0].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[0,0].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[0,0].set_ylabel('Pot [kW]')
ax[0,0].grid(zorder=0)
ax[0,0].set_xticks(range(0,25,4))

#Feb
i=1
hini=monthhours[i-1]
hend=monthhours[i]
ax[0,1].set_title(MONTHSTRING[i])
ax[0,1].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[0,1].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[0,1].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[0,1].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[0,1].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[0,1].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[0,1].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[0,1].grid(zorder=0)

#Mar
i=2
hini=monthhours[i-1]
hend=monthhours[i]
ax[0,2].set_title(MONTHSTRING[i])
ax[0,2].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[0,2].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[0,2].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[0,2].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[0,2].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[0,2].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[0,2].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[0,2].grid(zorder=0)

#Abr
i=3
hini=monthhours[i-1]
hend=monthhours[i]
ax[1,0].set_title(MONTHSTRING[i])
ax[1,0].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[1,0].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[1,0].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[1,0].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[1,0].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[1,0].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[1,0].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[1,0].set_ylabel('Pot [kW]')
ax[1,0].grid(zorder=0)

#May
i=4
hini=monthhours[i-1]
hend=monthhours[i]
ax[1,1].set_title(MONTHSTRING[i])
ax[1,1].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[1,1].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[1,1].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[1,1].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[1,1].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[1,1].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[1,1].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[1,1].grid(zorder=0)

#Jun
i=5
hini=monthhours[i-1]
hend=monthhours[i]
ax[1,2].set_title(MONTHSTRING[i])
ax[1,2].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[1,2].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[1,2].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[1,2].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[1,2].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[1,2].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[1,2].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[1,2].grid(zorder=0)

#Jul
i=6
hini=monthhours[i-1]
hend=monthhours[i]
ax[2,0].set_title(MONTHSTRING[i])
ax[2,0].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[2,0].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[2,0].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[2,0].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[2,0].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[2,0].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[2,0].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[2,0].set_ylabel('Pot [kW]')
ax[2,0].grid(zorder=0)

#Ago
i=7
hini=monthhours[i-1]
hend=monthhours[i]
ax[2,1].set_title(MONTHSTRING[i])
ax[2,1].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[2,1].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[2,1].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[2,1].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[2,1].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[2,1].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[2,1].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[2,1].grid(zorder=0)

#Sep
i=8
hini=monthhours[i-1]
hend=monthhours[i]
ax[2,2].set_title(MONTHSTRING[i])
ax[2,2].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[2,2].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[2,2].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[2,2].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[2,2].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[2,2].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[2,2].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[2,2].grid(zorder=0)

#Oct
i=9
hini=monthhours[i-1]
hend=monthhours[i]
ax[3,0].set_title(MONTHSTRING[i])
ax[3,0].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b')
ax[3,0].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[3,0].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[3,0].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[3,0].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[3,0].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[3,0].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[3,0].set_xlabel('Days')
ax[3,0].set_ylabel('Pot [kW]')
ax[3,0].grid(zorder=0)

#Nov
i=10
hini=monthhours[i-1]
hend=monthhours[i]
ax[3,1].set_title(MONTHSTRING[i])
ax[3,1].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b') 
ax[3,1].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[3,1].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[3,1].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[3,1].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[3,1].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[3,1].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[3,1].set_xlabel('Days')
ax[3,1].grid(zorder=0)

#Dic
i=11
hini=monthhours[i-1]
hend=monthhours[i]
ax[3,2].set_title(MONTHSTRING[i])
ax[3,2].plot(np.arange(1,25),np.median(Pdc_raw[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'b')
ax[3,2].plot(np.arange(1,25),np.median(Pac_net[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'k') 
ax[3,2].plot(np.arange(1,25),np.median(P_clipped[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'r') 
ax[3,2].plot(np.arange(1,25),np.median(P_shading[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'c') 
ax[3,2].plot(np.arange(1,25),np.median(P_loss[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'g') 
ax[3,2].plot(np.arange(1,25),np.median(P_sell[hini:hend].reshape(monthdays[i],24),axis=0)/1000,'m') 
ax[3,2].plot(np.arange(1,25),np.median(np.transpose(demand[:,hini:hend]).reshape(monthdays[i],24),axis=0)/1000,'k--') 
ax[3,2].set_xlabel('Days')
ax[3,2].grid(zorder=0)

plt.legend(['Pot DC Bruta','Pot AC neta','Vert por inver','Pérd por sombras','Pérd eléctricas','Pot vendida','Demanda'],loc='upper center', bbox_to_anchor=(-0.95, 8.5),
          ncol=3, fancybox=True, shadow=True)
plt.savefig("Fig3", dpi=600, bbox_inches="tight")
plt.show()

#%%
#Figure(4)
demand.shape=(1,demand.size)
plt.rcParams["figure.figsize"] = [12, 8]
fig, ax=plt.subplots(2,2, sharex=True, sharey=True)
fig.tight_layout(pad=1.2)

#Demanda
ax[0,0].set_title('Demanda eléctrica')
dummy=ax[0,0].contourf(np.transpose(demand.reshape(365,24))/1000, extent=[0, 365, 0, 24], cmap='jet')
ax[0,0].set_ylabel('Horas [-]')
ax[0,0].grid(zorder=0)
ax[0,0].set_yticks(range(0,25,4))
ax[0,0].set_xticks(range(0,365,50))
cbar=fig.colorbar(dummy, ax=ax[0,0])
cbar.set_label('Energía [kWh]', rotation=90)

#Generación AC
ax[0,1].set_title('Generación AC')
dummy=ax[0,1].contourf(np.transpose(P_sell.reshape(365,24)+Pac_net.reshape(365,24))/1000, extent=[0, 365, 0, 24], cmap='jet')
ax[0,1].grid(zorder=0)
ax[0,1].set_yticks(range(0,25,4))
cbar=fig.colorbar(dummy, ax=ax[0,1])
cbar.set_label('Energía [kWh]', rotation=90)

#Potencia vendida a la red
ax[1,0].set_title('Potencia vendida a la red')
dummy=ax[1,0].contourf(np.transpose(P_sell.reshape(365,24))/1000, extent=[0, 365, 0, 24], cmap='jet')
ax[1,0].set_ylabel('Horas [-]')
ax[1,0].set_xlabel('Days [-]')
ax[1,0].grid(zorder=0)
ax[1,0].set_yticks(range(0,25,4))
cbar=fig.colorbar(dummy, ax=ax[1,0])
cbar.set_label('Energía [kWh]', rotation=90)

#Pérdidas
ax[1,1].set_title('Pérdidas')
dummy=ax[1,1].contourf(np.transpose(P_loss.reshape(365,24)+P_shading.reshape(365,24)+P_clipped.reshape(365,24))/1000, extent=[0, 365, 0, 24], cmap='jet')
ax[1,1].set_xlabel('Days [-]')
ax[1,1].grid(zorder=0)
ax[1,1].set_yticks(range(0,25,4))
cbar=fig.colorbar(dummy, ax=ax[1,1])
cbar.set_label('Energía [kWh]', rotation=90)
plt.savefig("Fig4", dpi=600, bbox_inches="tight")
plt.show()

#%%
# Reporte final
print('--REPORTE--')
print('Pre-diseño eléctrico: N_paralelo=', N_parallel, 'y N_serie=', N_series)
print('Pre-diseño geométrico: ', N_ALONG_BOTTOM, ' paneles por fila y ', N_ROWS, ' filas')
print('Inversor seleccionado: ',inverter_names[inverterID], ' de ', Pmax_ac/1000, 'kW_ac' )
print('CAPEX= ',CAPEX/1e+6, ' MMCLP')
print('VAN= ',VAN/1e+6, ' MMCLP')
index = bisect_right(ROI, 0)+1
print('ROI en el año ',index)


