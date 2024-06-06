import warnings
import geopandas as gpd
from shapely.errors import ShapelyDeprecationWarning
import find_base_stations as antenna
import generate_users
import models
import objects.Params as p
import util

warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

# provinces = ['Drenthe', 'Flevoland', 'Friesland', 'Groningen', 'Limburg', 'Overijssel', 'Utrecht', 'Zeeland',
#              'Zuid-Holland', 'Gelderland', 'Noord-Brabant', 'Noord-Holland']
# municipalities = ['Middelburg', 'Maastricht', 'Groningen', 'Enschede', 'Emmen', 'Elburg',
#                   'Eindhoven', "'s-Gravenhage", 'Amsterdam', 'Almere']

# provinces = ['Overijssel', 'Friesland', 'Utrecht']
municipalities = ['Middelburg', 'Enschede', 'Amsterdam']
# municipalities = ['Amsterdam']

# provinces = ['Noord-Holland']
# municipalities = ['Amsterdam']

# MNOS = [['KPN'], ['T-Mobile'], ['Vodafone'], ['KPN', 'Vodafone', 'T-Mobile']]
# MNOS = [['KPN'], ['T-Mobile'], ['Vodafone']]
MNOS = [['KPN', 'Vodafone', 'T-Mobile']]
# MNOS = [['Vodafone', 'T-Mobile']]

fdp_per_MNO = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
fsp_per_MNO = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
fdp_per_MNO_fp = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
fsp_per_MNO_fp = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}

radius_disaster = 0  # 0, or a value if there is a disaster in the center of the region with radius
random_failure = 0 # 0.1  # BSs randomly fail with this probability
user_increase = 0  # an increase in number of users
back_up = True
# back_up = False

# sharing = ['T-Mobile', 'Vodafone']
sharing = MNOS[0]

radii = [500, 1000, 2500]
increases = [50, 100, 200]
#random = [0.05, 0.1, 0.25, 0.5]
#random = [0.1]
random = [0]


max_iterations = 2 #50

# technologies = None #[[util.BaseStationRadioType.NR]]
technologies = [[util.BaseStationRadioType.UMTS], [util.BaseStationRadioType.NR], [util.BaseStationRadioType.LTE]]
areas =  None #[util.AreaType.UMI]
# areas = [[util.AreaType.UMI], [util.AreaType.UMA], [util.AreaType.RMA]]
# fig, ax = plt.subplots()

zip_codes = gpd.read_file('data/square_statistics.shp')
# for random_failure in [0]:
# for area in areas:
for technology in [None]: #technologies:
    # print('Failure:', random_failure)
    # for province in provinces:
    for municipality in municipalities:
        j = 0
        for mno in MNOS:
            full_connections = {'KPN': {'KPN': [], 'T-Mobile': [], 'Vodafone': []},
                                'T-Mobile': {'KPN': [], 'T-Mobile': [], 'Vodafone': []},
                                'Vodafone': {'KPN': [], 'T-Mobile': [], 'Vodafone': []},
                                'no': {'KPN': [], 'T-Mobile': [], 'Vodafone': []}}
            data = []
            fdp, fsp, sat = [], [], []
            fdp_fp, fsp_fp, p_per_mno, p_per_mno_fp = [], [], [], []
            fdp_per_MNO, fsp_per_MNO = [], []
            fdp_per_MNO_fp, fsp_per_MNO_fp = [], []
            for iteration in range(max_iterations):
                print(iteration)
                # Retrieve zip code population + area data and make a region with specified zip codes
                # Columns are: aantal_inw (population), stedelijkh (urbanity), postcode (zip code), geometry,
                # popdensity (population density), municipali(city), scenario

                # Region that we want to investigate:
                # cities = util.find_cities(province)
                cities = [municipality]
                province = None

                percentage = 2 / 100  # percentage of active users

                seed = iteration # 

                params = p.Parameters(seed, zip_codes, mno, percentage, buffer_size=2000, city_list=cities,
                                      province=province, radius_disaster=radius_disaster, random_failure=random_failure,
                                      user_increase=user_increase, capacity_distribution=False, back_up=back_up,
                                      sharing=sharing, technology = technology, areas = areas)
                params = antenna.find_zip_code_region(params)

                #         # FINDING USERS
                params = generate_users.generate_users(params)
                #         # FINDING BSS
                params = antenna.load_bs(params)
                #         # FINDING LINKS
                with_power_control = True
                if with_power_control:
                    links, link_channel, snr, sinr, capacity, FDP, FSP, connections, power_per_MNO = models.find_links_new(params)
                else:
                    links, link_channel, snr, sinr, capacity, FDP, FSP, connections = models.find_links(params)
                    # links, link_channel, snr, sinr, capacity, FDP, FSP, interference_loss, connections = models.find_links_QoS(params)

                print(f'There are {params.number_of_bs} BSs and {params.number_of_users} users. Iter {seed}')
                if with_power_control:
                    FSP_fp = FSP[1]
                    FDP_fp = FDP[1]
                    power_per_MNO_fp = power_per_MNO[1]
                    FSP = FSP[0]
                    FDP = FDP[0]
                    power_per_MNO = power_per_MNO[0]
                    
                    p_per_mno_fp.append(power_per_MNO_fp)
                    for mno_aux in ['KPN', 'Vodafone', 'T-Mobile']:
                        print(f'POWER FP {mno_aux} = {power_per_MNO_fp[mno_aux]} W')

                    fraction_satisified_pop_fp = sum(FSP_fp) / params.number_of_users
                    fraction_disconnected_pop_fp = sum(FDP_fp) / params.number_of_users
                    fdp_fp.append(fraction_disconnected_pop_fp)
                    fsp_fp.append(fraction_satisified_pop_fp)
                    print(f'FDP FP = {fraction_disconnected_pop_fp}')
                    print(f'FSP FP = {fraction_satisified_pop_fp}')

                    p_per_mno.append(power_per_MNO)
                    for mno_aux in ['KPN', 'Vodafone', 'T-Mobile']:
                        print(f'POWER {mno_aux} = {power_per_MNO[mno_aux]} W')

                fraction_satisified_pop = sum(FSP) / params.number_of_users
                fraction_disconnected_pop = sum(FDP) / params.number_of_users

                print(f'FDP = {fraction_disconnected_pop}')
                print(f'FSP = {fraction_satisified_pop}')

                fdp.append(fraction_disconnected_pop)
                fsp.append(fraction_satisified_pop)

                fdp_per_MNO_aux = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
                fsp_per_MNO_aux = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
                fdp_per_MNO_fp_aux = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
                fsp_per_MNO_fp_aux = {MNO: list() for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
                for user in params.users:
                    MNO = user.provider
                    id = user.id
                    fdp_per_MNO_aux[MNO].append(FDP[id])
                    fsp_per_MNO_aux[MNO].append(FSP[id])
                    if with_power_control:
                        fdp_per_MNO_fp_aux[MNO].append(FDP_fp[id])
                        fsp_per_MNO_fp_aux[MNO].append(FSP_fp[id])

                fdp_per_MNO_aux = {MNO: sum(fdp_per_MNO_aux[MNO])/max(1,len(fdp_per_MNO_aux[MNO])) for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
                fsp_per_MNO_aux = {MNO: sum(fsp_per_MNO_aux[MNO])/max(1,len(fsp_per_MNO_aux[MNO])) for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
                fdp_per_MNO_fp_aux = {MNO: sum(fdp_per_MNO_fp_aux[MNO])/max(1,len(fdp_per_MNO_fp_aux[MNO])) for MNO in ['KPN', 'T-Mobile', 'Vodafone']}
                fsp_per_MNO_fp_aux = {MNO: sum(fsp_per_MNO_fp_aux[MNO])/max(1,len(fsp_per_MNO_fp_aux[MNO])) for MNO in ['KPN', 'T-Mobile', 'Vodafone']}

                fdp_per_MNO.append(fdp_per_MNO_aux)
                fsp_per_MNO.append(fsp_per_MNO_aux)
                fdp_per_MNO_fp.append(fdp_per_MNO_fp_aux)
                fsp_per_MNO_fp.append(fsp_per_MNO_fp_aux)
                """ #links, link_channel, snr, sinr, capacity, FDP, FSP, connections = models.find_links(params)
                # links, link_channel, snr, sinr, capacity, FDP, FSP, interference_loss, connections = models.find_links_QoS(params)
                #
                fraction_satisified_pop = sum(FSP) / params.number_of_users
                fraction_disconnected_pop = sum(FDP) / params.number_of_users
                # print(f'There are {params.number_of_bs} BSs and {params.number_of_users} users.')
                #
                # print(f'FDP = {fraction_disconnected_pop}')
                # print(f'FSP = {fraction_satisified_pop}')
                fdp.append(fraction_disconnected_pop)
                fsp.append(fraction_satisified_pop)

                # graph_functions.draw_graph(params, links, ax)
                for user in params.users:
                    MNO = user.provider
                    id = user.id
                    fdp_per_MNO[MNO].append(FDP[id])
                    fsp_per_MNO[MNO].append(FSP[id]) """

                for k, v in connections[0].items():
                    for i, j in v.items():
                        full_connections[k][i].append(j)
            
            for k, v in full_connections.items():
                for i, j in v.items():
                    #full_connections[k][i] = sum(j)/len(j)
                    full_connections[k][i] = 100*(sum(j)/len(j))/(params.number_of_users/3) #percent
            
            print(full_connections)
            util.to_data(fdp, f'data/Realisations/{params.filename}{max_iterations}_totalfdp.p')
            util.to_data(fsp, f'data/Realisations/{params.filename}{max_iterations}_totalfsp.p')
            if with_power_control:
                util.to_data(fdp_fp, f'data/Realisations/{params.filename}{max_iterations}_totalfdp_fp.p')
                util.to_data(fsp_fp, f'data/Realisations/{params.filename}{max_iterations}_totalfsp_fp.p')
                util.to_data(p_per_mno_fp, f'data/Realisations/{params.filename}{max_iterations}_totalpower_per_mno_fp.p')
                util.to_data(p_per_mno, f'data/Realisations/{params.filename}{max_iterations}_totalpower_per_mno.p')

                util.to_data(fdp_per_MNO, f'data/Realisations/{params.filename}{max_iterations}_fdp_per_MNO.p')
                util.to_data(fsp_per_MNO, f'data/Realisations/{params.filename}{max_iterations}_fsp_per_MNO.p')
                util.to_data(fdp_per_MNO_fp, f'data/Realisations/{params.filename}{max_iterations}_fdp_per_MNO_fp.p')
                util.to_data(fsp_per_MNO_fp, f'data/Realisations/{params.filename}{max_iterations}_fsp_per_MNO_fp.p')

        """ if type(MNOS[0]) == list and len(MNOS[0]) == 1:
            lijst = [MNOS[i][0] for i in range(len(MNOS))]
        elif len(MNOS[0]) > 1:
            lijst = MNOS[0]
        else:
            lijst = MNOS

        if province:
            print(province, ':', [sum(fdp_per_MNO[MNO]) / len(fdp_per_MNO[MNO]) for MNO in lijst])
            print(province, ':', [sum(fsp_per_MNO[MNO]) / len(fsp_per_MNO[MNO]) for MNO in lijst])
        else:
            print(municipality, ':', [sum(fdp_per_MNO[MNO]) / len(fdp_per_MNO[MNO]) for MNO in lijst])
            print(municipality, ':', [sum(fsp_per_MNO[MNO]) / len(fsp_per_MNO[MNO]) for MNO in lijst]) """

        """ util.to_data(fdp_per_MNO, f'data/Realisations/{params.filename}{max_iterations}_fdp_per_MNO.p')
        util.to_data(fsp_per_MNO, f'data/Realisations/{params.filename}{max_iterations}_fsp_per_MNO.p')
        if with_power_control:
            util.to_data(fdp_per_MNO_fp, f'data/Realisations/{params.filename}{max_iterations}_fdp_per_MNO_fp.p')
            util.to_data(fsp_per_MNO_fp, f'data/Realisations/{params.filename}{max_iterations}_fsp_per_MNO_fp.p') """

