# -*- encoding:utf-8 -*-
import osmnx as ox
import time
from shapely.geometry import Polygon
import os
from fmm import FastMapMatch, Network, NetworkGraph, UBODTGenAlgorithm, UBODT, FastMapMatchConfig, Edge, LineString
import pickle
import numpy as np


def save_graph_shapefile_directional(G, filepath=None, encoding="utf-8"):
    # default filepath if none was provided
    if filepath is None:
        filepath = os.path.join(ox.settings.data_folder, "graph_shapefile")

    # if save folder does not already exist, create it (shapefiles
    # get saved as set of files)
    if not filepath == "" and not os.path.exists(filepath):
        os.makedirs(filepath)
    filepath_nodes = os.path.join(filepath, "nodes.shp")
    filepath_edges = os.path.join(filepath, "edges.shp")

    # convert undirected graph to gdfs and stringify non-numeric columns
    gdf_nodes, gdf_edges = ox.utils_graph.graph_to_gdfs(G)
    gdf_nodes = ox.io._stringify_nonnumeric_cols(gdf_nodes)
    gdf_edges = ox.io._stringify_nonnumeric_cols(gdf_edges)
    # We need an unique ID for each edge
    gdf_edges["fid"] = gdf_edges.index
    # save the nodes and edges as separate ESRI shapefiles
    gdf_nodes.to_file(filepath_nodes, encoding=encoding)
    gdf_edges.to_file(filepath_edges, encoding=encoding)

# print("osmnx version",ox.__version__)

# Download by a bounding box
# bounds = (17.4110711999999985,18.4494298999999984,59.1412578999999994,59.8280297000000019) #
# x1,x2,y1,y2 = bounds
# boundary_polygon = Polygon([(x1,y1),(x2,y1),(x2,y2),(x1,y2)])


# for chengdu
boundary_polygon = Polygon([(104.043333, 30.727818), (104.129076,
                                                      30.726490), (104.129591, 30.655191), (104.042102, 30.652828)])
city = 'chengdu'
# print(boundary_polygon.boundary.xy)

# # for xian
# boundary_polygon = Polygon([(108.92309,34.279936),(109.008833,34.278608),(109.009348,34.207309),(108.921859,34.204946)])
# city = 'xian'

boundary_x = boundary_polygon.boundary.xy[0]
boundary_y = boundary_polygon.boundary.xy[1]
min_long = min(boundary_x)
max_long = max(boundary_x)
min_lat = min(boundary_y)
max_lat = max(boundary_y)

# print(min_long,max_long,min_lat,max_lat)

edge_one_hot_attributes = ['highway', 'oneway',
                           'lanes', 'bridge', 'junction', 'tunnel']
edge_float_attributes = ['length', 'maxspeed']
edge_linestring_attributes = ['coordinates']  # 直接用长度来表示特征即可


def downloadRoadNetwork():
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
    start_time = time.time()

    save_graph_shapefile_directional(G, filepath='../data/{}'.format(city))
    print("--- %s seconds ---" % (time.time() - start_time))


def generateEdgeNodeAttr():
    G = ox.graph_from_polygon(boundary_polygon, network_type='drive')
    start_time = time.time()
    nodes = {}
    edges = {}
    for k, v in G.nodes.items():
        nodes[k] = v

    for u, v, data in G.edges(nbunch=None, data=True):
        if 'geometry' in data.keys():
            geometry = data['geometry']
            x, y = geometry.xy
            data['coordinates'] = (list(x), list(y))
        edges[(u, v)] = data
    print(len(nodes))
    print(len(edges))
    with open('../data/{}'.format(city)+"/roadnetwork.pk", 'wb') as f:
        pickle.dump((nodes, edges), f)
    print("--- %s seconds ---" % (time.time() - start_time))
    with open('../data/{}'.format(city)+"/roadnetwork.pk", 'rb') as f:
        nodes, edges = pickle.load(f)
        print(len(nodes), len(edges))


def analyseRoadNetwork():
    nodes, edges = pickle.load(
        open('../data/{}'.format(city)+"/roadnetwork.pk", 'rb'))
    print(len(nodes), len(edges))
    # 分析每条边的性质
    # edge_one_hot_categories = {}
    # edge_floats = {}
    # for edge,data in edges.items():
    #     for k,v in data.items():
    #         if k in edge_one_hot_attributes:
    #             if type(v) is list:
    #                 v = '&'.join([str(i) for i in v])
    #             vs = edge_one_hot_categories.get(k,{})
    #             if v not in list(vs.keys()):
    #                 vs[v] = 1
    #             else:
    #                 vs[v] += 1
    #             edge_one_hot_categories[k] = vs
    #         elif k in edge_float_attributes:
    #             edge_floats[k] = edge_floats.get(k,0) + 1
    # for k,v in edge_one_hot_categories.items():
    #     print(k,len(v),v)
    # print(edge_floats)

    '''chengdu: 2575 5994
        highway 25 {'unclassified': 840, 'primary': 795, 'tertiary': 1463, 'secondary': 841, 'secondary_link': 49, 'residential': 1723, 'trunk': 59, 'trunk_link': 54, 'secondary&primary': 4, 'motorway&primary': 2, 'motorway_link': 4, 'secondary&tertiary': 1, 'primary_link': 51, 'tertiary_link': 27, 'living_street': 45, 'primary_link&tertiary': 1, 'residential&unclassified': 8, 'unclassified&tertiary': 12, 'residential&tertiary': 4, 'residential&living_street': 2, 'primary_link&primary': 3, 'residential&secondary': 2, 'trunk_link&primary_link': 1, 'secondary&unclassified': 2, 'secondary_link&secondary': 1}
        oneway 3 {False: 4175, True: 1818, 'False&True': 1}
        # name 721 {'昭觉寺南路': 26, '解放路一段': 10, '平福巷': 2, '一环路北四段': 14, '解放路二段': 4, '人民北路': 33, '武都路': 12, '新光华街': 9, '顺城大街': 37, '东城根南街': 8, '蜀都大道少城路': 1, '金盾路': 5, '文翁路': 5, '西玉龙街': 6, '东大街上东大街段': 11, '青石桥北街': 4, '署袜中街&青石桥南街': 2, '小南街': 11, '锦里中路': 5, '锦里滨河路': 6, '锦官桥&大石东路': 1, '芳邻路': 4, '锦里西路': 10, '蜀都大道人民西路': 22, '提督街': 14, '蜀都大道总府路': 8, '望平街': 6, '蜀都大道东风路&蜀都大道大慈寺路': 1, '三环路': 12, '凤凰山机场路': 6, '东御街': 3, '人民中路一段': 22, '东城根上街': 32, '东门街': 2, '羊市街': 8, '新华大道文武路': 9, '五丁路': 7, '一环路': 45, '白马寺街': 4, '五丁街': 4, '五丁路&五丁桥': 2, '西北桥河边街': 4, '九里堤南路': 12, '沙湾东一路': 12, '星月东街': 6, '九里堤北路': 9, '星科北街': 4, '成都市城北出口高速公路&中环路昭觉寺横路段': 2, '中环路昭觉寺横路段': 12, '跳蹬河南路': 4, '崔家店路': 5, '跳蹬河北路': 6, '崔家店路&万年路': 2, '府青路三段': 3, '府青路二段': 20, '府青路三段&中环路昭觉寺横路段': 1, '踏水桥北街': 6, '二环路东一段': 20, '光荣北路': 10, '沙湾路': 11, '三友路': 12, '马鞍北路': 2, '宁夏街': 7, '八宝街': 7, '花牌坊街': 6, '金沙路': 8, '为民路': 4, '下同仁路': 14, '蜀都大道通惠门路': 4, '双林路': 22, '双华路': 4, '蜀都大道金河路': 7, '长顺上街': 9, '将军街': 2, '桂花巷': 2, '北星大道一段': 15, '府青路一段': 13, '三槐树路': 7, '庆云街': 4, '红星路四段': 2, '书院南街': 6, '纱帽中街&纱帽北街': 2, '新鸿路': 23, '一环路东二段': 9, '玉双路': 13, '一环路东三段': 10, '府青立交': 15, '二环路东三段': 12, '东秀一路': 8, '染房街': 8, '西御街': 6, '交大路': 8, '群星路': 12, '泉水东路': 14, '中环路洞子口路段': 14, '韦家碾四路': 24, '中环路双荆路段': 25, '中环路昭觉寺横路段&中环路双荆路段': 1, 'v': 8, '八里桥路': 25, '荆竹南路': 12, '泉水路': 1, '赛云台东一路': 6, '蓉北商贸大道二段': 10, '站北东街': 8, '站北路': 6, '赛云台西一路': 4, '蓉北商贸大道一段': 16, '槐树街': 10, '长顺下街': 11, '西体路': 4, '金仙桥路': 4, '通锦桥路': 12, '西月城街': 4, '三洞桥路': 5, '西安北路': 7, '永陵路': 4, '实业街': 8, '西青路': 2, '西安南路': 7, '猛追湾街': 20, '天祥街': 4, '东安南路': 8, '望福街': 6, '华星路': 8, '华新路桥': 2, '锣锅巷': 3, '德盛路': 8, '康庄街': 2, '童子街': 2, '太升南路': 18, '西华门街': 14, '平安巷': 2, '西御河沿街': 2, '天成街': 2, '桂王桥西街': 4, '桂王桥北街': 4, '桂王桥南街': 1, '慈惠堂街': 2, '梓潼桥正街': 2, '布后街': 1, '隆兴街': 4, '纯阳观街': 4, '三倒拐街': 2, '忠烈祠东街': 4, '东玉龙街': 2, '拐枣树街': 4, '冻青树街': 8, '红星路': 18, '蓥华寺街&昭忠祠街': 1, '蓥华寺街': 2, '惜字宫南街': 4, '武城大街': 12, '庆云南街': 2, '祥和里': 2, '石油路': 6, '东安北路': 4, '玉双路&武城大街&武成门桥': 2, '蜀都大道东风路': 2, '东风路北一巷': 4, '小龙桥路': 10, '新鸿北支路': 4, '双林北横路': 10, '双林北支路': 14, '新鸿南一巷': 8, '建华南巷': 6, '建设南支路': 16, '新鸿南路': 8, '建设路': 16, '建设巷': 8, '建设南一路': 2, '建设南二路': 4, '马鞍东路': 4, '马鞍街': 4, '外曹家巷': 2, '马鞍西路': 2, '马鞍南路': 2, '东华门街': 14, '体育场路': 2, '梵音寺街': 6, '白丝街': 3, '西沟头巷': 2, '永兴巷': 2, '署袜北一街': 4, '梓潼桥西街': 4, '华兴上街': 1, '署袜北二街': 2, '兴隆街': 2, '交通路&横九龙巷': 2, '交通路': 2, '古卧龙桥街': 4, '学道街': 4, '青石桥中街': 2, '走马街': 4, '文庙前街': 4, '上汪家拐街': 4, '文庙西街': 6, '二环路': 10, '九里堤中路': 7, '九里堤南支路': 6, '沙洲街': 2, '沙湾东二路': 4, '圃园南二路': 4, '圃园路': 2, '圃园南一路': 4, '圃园北路': 2, '建设北路一段': 8, '桃蹊路': 10, '双建路': 22, '桃溪巷': 4, '怡福路': 4, '一环路东一段': 13, '亚光路': 4, '国光路': 4, '宏明路': 2, '星辉东路': 6, '星辉中路': 2, '太升北路&马鞍南路&太升桥': 2, '张家巷': 4, '恒德路': 2, '花圃路': 4, '前锋路': 6, '虹波路': 12, '华油路': 8, '平福路': 8, '解放西路': 18, '互助路': 4, '二环路北三段': 24, '北站东二路': 3, '肖家村四巷': 4, 'Beizhan West 1st Lane': 12, '北站西二路': 7, '北站西一路': 8, '北站东一路': 10, '西安中路一巷': 8, '青羊东二路': 2, '枣子巷': 6, '金沙北一路': 8, '四道街': 4, '上同仁路': 8, '西二道街&三道街': 2, '过街楼街': 4, '同心路': 6, '西大街': 4, '中同仁路': 10, '通顺桥街': 2, '楞伽庵街': 2, '白云寺街': 2, '金丝街': 6, '红石柱街': 2, '金马街': 2, '白家塘街': 4, '北东街': 4, '酱园公所街': 2, '草市街': 6, '成华街': 4, '成华南街': 6, '星辉西路': 10, '成华西街': 10, '白马寺北顺街': 2, '肖家村三巷': 2, '文家后巷': 2, '烟袋巷': 1, '文庙后街': 6, '下汪家拐街': 2, '横陕西街': 2, '陕西街': 10, '君平街': 4, '双林路&新华桥&三槐树路': 2, '天祥滨河路': 1, '中新街&南新街': 2, '向荣桥街': 2, '新半边街&老半边街': 2, '西糠市街': 2, '新半边街': 2, '三多里': 6, '新鸿南支路': 4, '正科甲巷': 4, '上翔街': 2, '星河路': 2, '星汉路': 22, '交桂路': 6, '星科路': 6, '星辰路': 6, '星汉北路': 2, '长月路': 2, '建设南三路': 6, '人民南路一段': 6, '青龙场立交桥': 7, '驷马桥路': 10, '中环路八里庄路段': 7, '刃具立交': 6, '二环路东四段': 1, '踏水桥西街': 8, '二环路北四段': 8, '秀苑路': 8, '华安街': 2, '游乐园滨河路': 5, '府青路一段&府青立交': 1, '猛追湾滨河路': 1, '万年路': 10, '积步街': 2, '红照壁街': 5, '西御街&东御街': 1, '人民西路': 1, '南灯巷': 2, '忠孝巷': 4, '蓉都大道': 1, '先锋路&顺沙巷': 2, '横桥街': 18, '双沙东路': 16, '双瑞一路': 6, '荆竹坝路': 16, '沙荆路': 4, '羊子山西路': 12, '仁义路': 6, '云龙路': 8, '花径路': 10, '福祥街': 2, '赛云台东二路': 6, '站北中街': 12, '赛云台北支路': 10, '富康街': 4, '东沙街': 4, '升仙湖北路': 12, '水武街': 4, '站北北街': 8, '五福桥东路': 12, '红花西路': 2, '白下路': 4, '文殊院街': 4, '荆竹西路': 11, '五块石路': 10, 'Wufuqiao Road East': 2, '西北桥北街': 6, '斌升街': 2, 'West Madao Street': 6, '福善巷': 2, '大安西路': 4, '建业路': 10, '建设南路': 20, '建兴路': 12, '秀苑东路': 10, '秀苑桥': 2, '王贾路': 24, '泰宏路': 12, '红花东路': 4, '田家巷': 5, '泰宏街': 4, '太升路沙河桥&三友路': 2, '树蓓街': 12, '福蓓街': 2, '新怡路': 6, '怡福巷': 4, '建设北路三段': 17, '建和路': 4, '东珠市街': 4, '北大街': 4, '西珠市街': 2, '灶君庙街': 4, '玉皇观街': 4, '小关庙后街': 2, '大安东路&华星路': 2, '大安东路': 4, '东较场街': 12, '昭忠祠街': 4, '五昭路': 2, '建设北街': 4, '踏水桥&建设北路二段': 2, '建设支巷': 6, '建设中路': 4, '猛追湾横街': 6, '培华西路': 2, '培华东路': 4, '红光路': 4, '建设南二路&建设南路&建设南三路': 2, '猛追湾东街': 6, '建设南街': 4, '杉板桥南一路': 6, '东秀二路': 6, '西胜街': 4, '建祥路': 10, '功博路': 6, '二仙桥西一巷': 4, '双建北巷': 6, '文德路': 24, '香木林路&桃溪巷': 2, '香木林路': 6, '新风巷': 2, '明珠路': 4, '香木林路&明珠路': 2, '建设北路二段': 9, '光明滨河路': 4, '曲水路': 6, '星辉桥': 2, '五世同堂街': 2, '城隍庙街': 2, '岳府街': 9, '双栅子街': 2, '赤虎桥东路': 20, '洪山路': 14, '东篱路': 2, '蜀都大道人民东路': 6, '二环高架路': 18, '成彭立交': 1, '东荆路': 18, '南城塘坎街': 6, '羊子山路': 6, '青龙街': 5, '青龙巷': 6, '头福街': 2, '五岳宫街': 2, '树蓓巷': 4, '东秀二路&杉板桥南三路': 2, '杉板桥路': 17, '旭光路': 2, '东郊记忆南路': 4, '圣灯路': 14, '建功路': 8, '万年场横街': 2, '大安东路桥': 2, '星辉东滨河路': 1, '红星路&红星桥': 2, '二仙桥北路': 5, '星辰东二街': 6, '站西路': 6, '玉居庵东路': 6, '站西桥东巷': 2, '东马道街': 4, '金家坝街': 2, '珠宝街': 2, '文殊院巷': 2, '天成街&大福建营巷': 2, '西北桥东街': 2, '府河桥西路': 14, '府河桥巷': 2, '西体北路': 8, '万担仓路': 4, '新村河边街': 2, '万福桥&人民北路': 2, '新华大道江汉路': 10, '洛阳路': 2, '青龙路': 17, '荆竹东路': 9, '荆竹中路': 15, '青冈北路': 6, '青冈南路': 6, '崔家店北一路': 13, '崔家店北二路': 8, '地勘路': 4, '民兴四路': 8, '中环路二仙桥西路段': 6, '庆云西街': 4, '落虹桥街': 2, '四圣祠北街': 2, '九里堤西路': 1, '站前路': 3, '北顺城街': 2, '东顺城中街': 2, '水东门街': 4, '中道街': 2, '天涯石南街': 2, '天涯石北街': 6, '四圣祠南街': 4, '竹林巷': 2, '王家塘街': 4, '商业街': 2, '长顺中街': 9, '西林路': 16, '洪山南路': 6, '洪山北路': 18, '荆翠西路': 16, '荆翠南三街': 2, '荆顺路': 6, '荆翠东路': 20, '东紫路': 12, '东丽街': 4, '韦家碾二路': 6, '韦家碾一路': 10, '韦家碾三路': 10, '文和路': 2, '文庙西街&文和路': 2, '包家巷': 10, '方池街': 2, '蜀华街': 4, '横小南街': 4, '井巷子': 2, '赛云台南路': 4, '站北西横街': 6, '致兴路': 12, '致强街': 6, '致顺南二街': 6, '致顺路': 12, '致强环街': 6, '昭青路': 6, '致兴二路': 14, '致力路': 8, '东裕路&致力路': 2, '福兴街': 1, '华兴东街': 1, '华兴正街': 2, '红庙子街': 4, '锣锅巷&玉带侨街': 1, '东打铜街': 6, '玉沙路': 10, '太升北路': 10, '东通顺街': 2, '正通顺街': 2, '兴禅寺街': 2, '玉泉街': 4, '鼓楼南街': 2, '古中市街': 2, '鼓楼洞街': 2, '大墙西街': 6, '鼓楼北一街': 6, '鼓楼北二街': 4, '忠烈祠西街': 3, '东城拐下街': 2, '三桂前街': 2, '方正东街': 4, '正府街': 6, '七家巷': 2, '福德街': 4, '鼓楼北四街': 2, '新开寺街': 2, '狮子巷': 2, '小关庙街': 2, '方正街': 2, '帘官公所街': 2, '马镇街': 4, '石马巷': 2, '鼓楼北三街': 4, '梓潼街': 2, '仁厚街': 2, '多子巷': 2, '支矶石街': 4, '九里堤东路&九里堤西路': 1, '政通路': 10, '东二道街': 2, '北站西二巷&肖家村三巷': 2, '北站西二巷': 2, '玉赛路': 6, '五块石东一路': 4, '星辰东一街': 4, '干槐树街': 4, '书院西街': 2, '天涯石东街': 2, '站北东横街': 6, '五块石西路': 2, '和平街': 4, '赛云台北路': 6, '王家巷': 4, '林巷子': 4, '老东城根街': 6, '西府北街': 2, '商业后街': 4, '长发街': 2, '栅子街&小通巷': 2, '东胜街': 2, '云龙南路': 2, '致祥路': 4, '奎星楼街': 4, '吉祥街': 4, '西马棚街': 2, '红墙巷': 2, '东马棚街': 2, '横东城根街': 2, '西府北街&正府街': 2, '东御河沿街': 4, '青石桥南街': 4, '向阳街': 2, '府青跨线桥': 2, '北较场西路': 12, '中环路八里庄路段&青龙场立交桥': 1, '西林街': 12, '西林二街': 6, '西林三街': 4, '西林四街&西林街': 2, '西林四街': 4, '灯笼街': 2, '金丰大道': 1, '中环路金府路段': 4, '锦兴路': 2, '青年里': 2, '致和路': 2, '熊猫大道': 5, '金丰高架桥': 1, '致兴三路': 8, '青龙场新街': 6, '二仙桥海滨湾支路': 2, '东荆路&致强街': 2, '玉带侨街': 1, '祠堂街': 1, '万和路': 8, '万和路&北较场西路': 2, '焦家巷': 4, '蜀都大道十二桥路': 4, '文武路': 1, '锦官桥': 1, '大石东路': 2, '和田路': 6, '倒桑树街': 2, '南浦西路': 2, '通惠门路': 2, '柿子巷': 2, '五福桥路&赛云台西一路': 2, '蓉北商贸大道二段&蓉北商贸大道一段': 1, '玉居庵东路&玉赛路': 2, '玉居庵西路': 2, '赛云台西二路': 2, '杉板桥南三路': 4, '龙绵街': 18, '东裕路': 10, '东泰路': 12, '大安滨河路': 4, '东大街': 1, '东顺城南街': 6, '总府路': 1, '东林三路': 6, '荆翠东路&龙绵街': 2, '杉板桥南五路': 12, '崔家店横二街': 4, '东风路北二巷': 4, '北门大桥&解放路二段&北大街': 2, '中环路二仙桥东路段': 1, '东林一路': 12, '蜀都大道人民西路&蜀都大道总府路': 2, '署袜中街': 6, '北新街': 2, '蜀都大道大慈寺路': 5, '三环路辅道': 5, '仁爱路': 8, '杉板桥南四路': 8, '洁美街&积步东巷': 2, '崔家店横一街': 4, '积步东巷': 2, '云龙南一街': 6, '小河街': 2, '公交路': 1, '北打铜街': 2, '蜀龙路': 4, '荆竹坝西街': 4, '荆顺西街': 2, '双富路': 6, '双水碾街': 2, '双兴街': 4, '双富路&双水碾街': 2, '双耀二路': 8, '升仙湖南路': 2, '双耀一路': 8, '凤凰山高架路': 6, '凤凰山高架': 3, '天仙桥北路': 8, '爵版街': 2, 'Chuan Lane': 2, '书院东街': 2, '布坝子街': 2, '新鸿南二巷': 6, '双林巷': 10, '建华北巷': 2, '三友巷': 2, '马鞍山路': 2, '马王庙街': 2, '铜丝街': 4, '光明巷': 2, '内姜街': 2, '昭忠祠街&蓥华寺街': 1, '双桥路': 5, '双桥路南六街': 2, '菽香里一巷': 4, '菽香里二巷&建设南一路': 2, '菽香里': 4, '菽香里二巷': 2, '建中路': 4, '天仙桥滨河路': 2, '城守街': 2, '中新街': 6, '三圣祠街': 2, '中新横街': 2, '新街后巷子': 6, '锦华馆': 2, '南沟头巷': 2, '横九龙巷': 2, '梨花街': 8, '上半节巷': 2, '泡桐树街': 2, '上南大街': 1, '星月西街': 6, '星光街': 2, '交桂二巷': 8, '交桂巷': 2, '交桂三巷': 2, '顺星街': 2, '横过街楼街': 2, '蜀都大道人民西路&蜀都大道金河路': 1, '大安中路': 3, '大树拐街': 2, '驷马桥二路': 4, '民兴二路': 8, '仙韵一路': 2, '二仙桥北一路': 6, '东新街': 2, '府青东街': 2, '西体路&五丁桥&武都路': 1, '武都路&五丁桥&西体路': 1, '蜀都大道十二桥路&蜀都大道通惠门路': 1, '杉板桥南二路': 2, '通锦桥路&新华大道江汉路': 2, '桂王桥东街': 1, '二仙桥北路&蜀龙路': 1, '民兴三路': 8, '枫丹路': 5, '青羊东二路二巷': 2, '致力路下穿隧道&致力路': 1, '鼓楼北一街二巷子': 2, '二环路北三段&二环路北四段': 1, '高车一路': 6, '东林二路&青龙路': 1, 'Funing Road': 6, '九里堤东路': 1, '中环路金府路段&中环路洞子口路段': 1}
        lanes 7 {'3': 455, '4': 88, '2': 201, '1': 33, '3&2': 1, '2&4': 1, '3&4': 2}
        bridge 3 {'yes': 278, 'viaduct': 30, 'yes&viaduct': 1}
        # ref 5 {'S57': 2, '中环路': 33, '二环路': 12, '羊西线': 6, 'S105': 7}
        junction 1 {'roundabout': 20}
        tunnel 1 {'yes': 14}
        {'length': 5994, 'maxspeed': 101}
    '''

    # 只有部分边有最大速度，分析一下是每种道路的最大速度，用于估计最小花费的时间
    # roadMaxSpeed = {}
    # for edge,data in edges.items():
    #     speed = None
    #     for k,v in data.items():
    #         if k=='maxspeed':
    #             speed = v
    #     if speed is not None:
    #         v = data['highway']
    #         if type(v) is list:
    #             v = '&'.join([str(i) for i in v])
    #         hi_speed = roadMaxSpeed.get(v,[])
    #         hi_speed.append(speed)
    #         roadMaxSpeed[v] = hi_speed
    # print(roadMaxSpeed)

    # 分析路网路口节点的信息
    # print(nodes)
    # {288416374: {'y': 30.7147948, 'x': 104.100294, 'osmid': 288416374}}
    # print(edges)
    # (2041337968, 1885342916): {'osmid': 193613865, 'name': '秀苑东路', 'highway': 'residential', 'oneway': False, 'length': 549.625, 'geometry': <shapely.geometry.linestring.LineString object at 0x11d93b080>, 'coordinates': ([104.0998961, 104.1012453, 104.1020891, 104.1025075, 104.1027114, 104.1027259, 104.1026498, 104.1026472], [30.680099, 30.679723, 30.6794988, 30.6792496, 30.6789359, 30.6783297, 30.6768227, 30.676771])}


def parseMaxSpeed():
    file = '../data/roadMaxSpeed'
    r_d = {}
    with open(file, 'r', encoding='utf-8') as r:
        for line in r.readlines():
            data = line.strip().split(' ')
            print(data)
            # assert len(data)==3
            key = data[0][1:-1]
            value = float(data[-1])
            r_d[key] = value
    print(r_d)
    return r_d

# 加载路网的节点和边以及它们的固定属性，每条边都要有每个categorical/numerical特征，没有的赋予默认值; 每条边都要有节点经纬度的相对位置，节点的入读、出度


def loadGraph():
    nodes, edges = pickle.load(
        open('../data/{}'.format(city) + "/roadnetwork.pk", 'rb'))
    print(len(nodes), len(edges))
    node_ids = list(nodes.keys())
    node_res = {}
    output_edge_dict = {}
    input_edge_dict = {}
    for e in edges.keys():
        assert e[0] in node_ids and e[1] in node_ids
        input_list = input_edge_dict.get(e[1], [])
        input_list.append(e[0])
        input_edge_dict[e[1]] = input_list

        output_list = output_edge_dict.get(e[0], [])
        output_list.append(e[1])
        output_edge_dict[e[0]] = output_list

    for k, v in nodes.items():
        # features = []
        long, lat = v['x'], v['y']
        assert long >= min_long and long <= max_long
        assert lat >= min_lat and lat <= max_lat
        features = [(long-min_long)/(max_long-min_long),
                    (lat-min_lat)/(max_lat-min_lat),
                    len(input_edge_dict.get(k, [])),
                    len(output_edge_dict.get(k, []))]
        node_res[k] = features

    # print(node_res)
    edge_res = {}
    default_max_speed = parseMaxSpeed()
    one_hot_categories = {}
    for attr in edge_one_hot_attributes:
        for _, data in edges.items():
            v = data.get(attr, None)
            vs = one_hot_categories.get(attr, [])
            if type(v) is list:
                for v1 in v:
                    if v1 not in vs:
                        vs.append(v1)
            # elif v is None:
            #     vs.insert(0,None)
            else:
                if v not in vs:
                    if v is None:
                        vs.insert(0, None)
                    else:
                        vs.append(v)
            one_hot_categories[attr] = vs

    print(one_hot_categories)

    for e, data in edges.items():
        # print(node_res[e[0]])
        features = node_res[e[0]][:2] + node_res[e[1]][:2]  # 首尾点的经纬度，定位了边的地理坐标
        # 边的one-hot属性
        for attr in edge_one_hot_attributes:
            all_v = one_hot_categories[attr]
            feature = [0 for i in range(len(all_v))]
            v = data.get(attr, None)
            if type(v) is list:
                for v1 in v:
                    i = all_v.index(v1)
                    feature[i] = 1
            else:
                i = all_v.index(v)
                feature[i] = 1
            features += feature
        # 边的float值属性
        for attr in edge_float_attributes:
            v = data.get(attr, None)
            if v is None and attr is 'maxspeed':
                road_type = data['highway']
                default_speed = np.mean([default_max_speed[r] for r in road_type]) if type(
                    road_type) is list else default_max_speed[road_type]
                features.append(float(default_speed)/100)
            elif v is not None:
                features.append(
                    float(v)/100 if attr is 'maxspeed' else float(v)/1000)
            else:
                raise Exception(
                    "Sorry, there is no valid value for the attribute-{} of edge-{}".format(attr, e))

        edge_res[e] = features

    return node_res, edge_res, input_edge_dict, output_edge_dict


def mapmatching():
    network = Network('../data/{}'.format(city)+"/edges.shp", "fid", "u", "v")
    print("Nodes {} edges {}".format(
        network.get_node_count(), network.get_edge_count()))
    graph = NetworkGraph(network)
    # graph.print_graph()
    # print(graph.get_node_id(0))
    for i in range(network.get_edge_count()):
        e = graph.get_edge(i)
        print(e.id, e.index, e.source, e.target, e.length)
        g = e.geom
        print(g.export_json())


# downloadRoadNetwork()

# sf = shapefile.Reader('../data/{}'.format(city)+"/edges.shp")
# sf = shapefile.Reader('../data/{}'.format(city)+"/nodes.shp")
# shapes = sf.shapes()
# print(len(shapes))
# for shape in shapes:
#     print(shape.shapeType, shape.shapeTypeName, shape.points, shape.parts)
# generateEdgeNodeAttr()
# analyseRoadNetwork()
# parseMaxSpeed()
node_res, edge_res, input_edge_dict, output_edge_dict = loadGraph()
# print(node_res)
for e, v in edge_res.items():
    print(e, v)

# mapmatching()
