import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

__all__ = ['plot_radial', 'plot_pca', 'plot_tsne', 'plot_umap', 'quick_fitted_tree']

def plot_radial(clf, X=None,Y=None, data=None, feature_cols=None, label_col=None,
                num_samples=100, levels=None,edges_labels=None, draw_labels=None,
                style='radplot', bbox='dark', cmap='pairs', tree_node_size=50,
                leaf_node_size=50, node_size=50, l_width=1, l_alpha=1,
                fig_res=72, save_img=False, img_res=300, png_transparent=True,
                spring=False, smooth_edges=False,smooth_d=None, smooth_res=50, random_state=None) :

    """Draw radial plot.
    feature_cols
    Parameters
    ----------
    clf : sklearn.tree.DecisionTreeClassifier
        Decision tree model to get the paths and tree structure

    X, Y, data : pandas.DataFrame or np.ndarray
        Data to test the tree and plot radial.
        Either 'data' or 'X' and 'Y' must be provided.

    feature_cols : list, optional
        List of features to select from 'data'. If None, data.columns[:-1] will be selected.

    label_col : list, optional
        List of features to select from 'data'. If None, data.columns[-1] will be selected.

    num_samples : int, default 100
        Number of samples to select from data. Please note that the processing time will increase by num_samples^2.

    levels : pd.DataFrame, optional
        DataFrame with one row (inxex=0) with a list of categorical levels for each column in the data.

    edges_labels : list, optional
        List of pretty labels to replace column names in the plot (default=None)

    draw_labels : bool, default True
        Draw edges labels

    save_img : bool, default True
        Save PNG file in /plots/ folder.

    style : str, default radplot
        Set matplotlib style options.

    bbox : Dict, default {'boxstyle':'round', 'ec':'k', 'fc':'k', 'alpha':0.8}
        bbox style parameters for edges labels.

    cmap : str or list, default [red, magenta, blue, cyan, green, yellow]
        cmap string or list of hexcolors.
        Even indexes will color each correctely predicted class and odds will color the wrong ones.

    tree_node_size : int, -- under development --
        Size of tree nodes.

    leaf_node_size : int, -- under development --
        Size of leaf nodes.

    node_size : int, default 50
        Size of all tree nodes.

    l_width : float, default 1
        Edges lines width.

    l_alpha : float, default .1
        Edges lines alpha

    spring : bool, default False
        Apply tension on tree branches.

    smooth_edges : bool, default False
        Replace branches lines by paths splines.

    smooth_d : int, optional
        Degree of spline derivative.
        Activate smooth_edges.

    smooth_res : int, default 50
        Number of segments of the resulting spline.

    random_state : int, optional
        Seed for the random number generator.

    """
    from sklearn import tree, model_selection, feature_selection
    import networkx as nx
    from tqdm import tqdm_notebook as tqdm

    def make_graph(X, Y, clf, feature_cols, levels_0, tree_node_size=100, leaf_node_size=20, label_hex_colors=None, feat_short=None) :
        lhex = label_hex_colors
        t = clf.tree_
        x = X.copy()
        cat_cols = x.select_dtypes(('object', 'category')).columns
        if len(cat_cols) :
            x[cat_cols] = x[cat_cols].astype('category').apply(lambda col : col.cat.codes)
        y = Y.copy()
        num_classes = int(y.max()[0] + 1)
        x.index = range(t.node_count, t.node_count+len(x.index))
        y.index = range(t.node_count, t.node_count+len(x.index))
        G = nx.Graph()
        labels = get_edges_labels(X, Y, clf, feature_cols, levels_0=levels_0, feat_short=feat_short)
        ginis = (t.impurity - t.value.argmax(axis=2)[:,0]/2)+.5
        for i in range(t.node_count) :
            G.add_node(i, gini= ginis[i], node_size=tree_node_size, samples= t.weighted_n_node_samples[i], value= t.value[i])
        for i, f in x.iterrows() :
            G.add_node(i, node_size=leaf_node_size, **f.to_dict())
        for i in range(t.node_count) :
            i_l = t.children_left[i]
            i_r = t.children_right[i]
            if i_l != -1 :
                g = ginis[i_l].astype(int)
                label = labels[(i,i_l)]
                if len(label) == 0 :
                    label = None
                v = t.value[i_l,0]
                c = mpl.colors.to_hex(((v/v.sum()) * label_hex_colors[range(0,num_classes*2,2)].T).T.sum(axis=0))
                G.add_edge(i, i_l, weight=G.nodes[i_l]['samples'], color=c, label=label)
            if i_r != -1 :
                g = ginis[i_r].astype(int)
                label = labels[(i,i_r)]
                if len(label) == 0 :
                    label = None
                v = t.value[i_r,0]
                c = mpl.colors.to_hex(((v/v.sum()) * label_hex_colors[range(0,num_classes*2,2)].T).T.sum(axis=0))
                G.add_edge(i, i_r, weight=G.nodes[i_l]['samples'], color=c, label=label)
            v = t.value[i,0]
            c = mpl.colors.to_hex(((v/v.sum()) * label_hex_colors[range(0,num_classes*2,2)].T).T.sum(axis=0))
            G.nodes[i]['color'] = c
        p = pd.DataFrame([np.argwhere(path_)[-1][0] for path_ in clf.decision_path(x).toarray()])
        p.index = x.index
        p.columns = ['parent_n']
        p['loss'] = y.values[:,0] - t.value[p.values[:,0]].argmax(axis=2)[:,0]
        p = p.sort_values(by=['parent_n','loss'])
        p['pos'] = np.linspace(0,1,len(p), endpoint=False)
        p = p.sort_index()
        for j, n in enumerate(clf.decision_path(x).toarray()) :
            node_num = x.index[j]
            parent_n = np.argwhere(n)[-1][0]
            y_pred = t.value[parent_n].argmax()
            y_labl = int(y.iloc[j][0])
            loss = y_labl-y_pred
            loss = np.abs(loss)
            c = mpl.colors.to_hex(label_hex_colors[y_labl*2 + bool(loss)])
            G.add_edge(parent_n, node_num, weight=1, color=c)
            G.nodes[node_num]['color'] = c
            G.nodes[node_num]['loss'] = y_labl-y_pred
            G.nodes[node_num]['pos'] =  (1j)**(p.loc[node_num,'pos']*4)
            G.nodes[node_num]['depth'] = t.max_depth+1
        return G, p, x.index

    def get_edges_labels(X, Y, clf, feature_cols, levels_0=None, feat_short=None) :
        t = clf.tree_
        levels = levels_0
        levels.loc[1] = levels.apply(lambda x : [[int(min(*x).left),int(max(*x).right)]] if isinstance(x[0][0],pd.Interval) else [[min(*x),max(*x)]], axis=0).loc[0]
        levels.loc[2] = levels.loc[1]
        levels.loc[3] = levels.loc[0]
        levels.loc[4] = levels.loc[0]
        if feat_short == None :
            feat_short = [s.lower()[:5] for s in feature_cols]
        feat = t.feature
        feat_short = np.array(feat_short)[feat]
        feat = np.array(feature_cols)[feat]
        threshold = t.threshold.copy()

        def get_each(i=0, labels = {}, levels=levels) :
            l_node = t.children_left[i]
            r_node = t.children_right[i]
            if l_node == r_node :
                return labels
            l_split = levels.loc[1,feat[i]].copy()
            r_split = levels.loc[2,feat[i]].copy()
            l_idx = int(threshold[i])
            r_idx = int(threshold[i]) + 1
            if isinstance(levels.loc[0,feat[i]][0], int) :
                l_split[1] = l_idx
                r_split[0] = r_idx
            if isinstance(levels.loc[0,feat[i]][0], float) :
                l_split[1] = round(l_idx,1)
                r_split[0] = round(r_idx,1)
            feat_levels = np.array(levels.loc[0,feat[i]])
            if isinstance(levels.loc[0,feat[i]][0], pd.Interval) :
                l_split[1] = feat_levels[l_idx].right.astype('int')
                r_split[0] = feat_levels[r_idx].left.astype('int')
            levels.loc[1,feat[i]] = l_split
            levels.loc[2,feat[i]] = r_split
            if isinstance(levels.loc[0,feat[i]][0], str) :
                l_to_split = feat_levels[:r_idx]
                r_to_split = feat_levels[r_idx:]
                l_split = np.array(levels.loc[3,feat[i]])
                r_split = np.array(levels.loc[4,feat[i]])
                l_split = list(set(l_split) - set(r_to_split))
                r_split = list(set(r_split) - set(l_to_split))
                levels.loc[3,feat[i]] = l_split
                levels.loc[4,feat[i]] = r_split
                # l_split = ' '.join(map(str, l_split))
                # r_split = ' '.join(map(str, r_split))
            l_label = f'{feat_short[i]} [{" | ".join(map(str,l_split))}]'
            r_label = f'{feat_short[i]} [{" | ".join(map(str,r_split))}]'
            labels[(i, l_node)] = l_label
            labels[(i, r_node)] = r_label
            labels = get_each(i=l_node, labels=labels, levels=levels)
            labels = get_each(i=r_node, labels=labels, levels=levels)
            return labels

        labels = get_each()
        return labels

    def get_depth(G, clf, node_paths) :
        t = clf.tree_
        p = node_paths
        depths = nx.shortest_path_length(G,0)
        a = pd.DataFrame(list(depths.items()), columns=['node', 'depth'])
        a = a.set_index('node')
        pos_leafs = p[['parent_n','pos']].groupby(['parent_n']).mean()
        a.loc[:,'pos'] = pos_leafs
        a.loc[:,'depth_r'] = 0
        mask = a.index.isin(np.argwhere(t.children_left == -1).flatten()) | (a['depth'] >= t.max_depth)
        a.loc[mask] = a.sort_index().loc[mask].interpolate().fillna(method='bfill').fillna(method='ffill')
        a.loc[a['pos'].notna(),'depth_r'] = t.max_depth
        a.loc[(a['depth'] == t.max_depth),'depth_r'] = t.max_depth
        a.loc[p.index,'depth_r'] = t.max_depth +1
        a.loc[p.index,'pos'] = p['pos']
        i = t.max_depth
        while a['pos'].isna().any() :
            mask = (a['depth'] >= i) | (a['pos'].notna())
            a.loc[mask,'pos'] = a.loc[mask,'pos'].sort_index().interpolate()
            a.loc[(a['depth'] >= i) | (a['pos'].notna()),'pos'] = a.loc[(a['depth'] >= i) | (a['pos'].notna()),'pos'].fillna(0.0001)
            i = i - 1
            mask = a.loc[(a['pos'].isna())].index
            mean_pos = (a.loc[t.children_left[mask],'pos'].values + a.loc[t.children_right[mask],'pos'].values)/2
            a.loc[(a['pos'].isna()), 'depth_r'] = i
            a.loc[(a['pos'].isna()), 'pos'] = mean_pos.tolist()
            if i < -2 : # to avoid an infinite loop
                print('Error in line 239 radtree/__init__.py : get_depth() while loop')
                break
        df_depths = a.sort_index()
        node_paths = p
        return df_depths, node_paths

    def make_radial_plot(G,X,Y, clf, df_depths, node_paths, leaf_nodes,
                         draw_labels=None, bbox=None, node_size=None,
                         l_width=1, l_alpha=1, smooth_res=50, smooth_d=3, spring=True,
                         smooth_edges=False, edges_labels=None, bg_color="#00000F",
                         fig_res=72, save_img=True, img_res=72, png_transparent=True) :
        t = clf.tree_
        x = X.copy()
        cat_cols = x.select_dtypes(('object', 'category')).columns
        if len(cat_cols) :
            x[cat_cols] = x[cat_cols].astype('category').apply(lambda col : col.cat.codes)
        y = Y.copy()
        x.index = range(t.node_count, t.node_count+len(x.index))
        y.index = range(t.node_count, t.node_count+len(x.index))
        a = df_depths
        p = node_paths
        if draw_labels == None :
            draw_labels = not smooth_edges
        paths_a = nx.shortest_path_length(G,0)
        paths_a = pd.DataFrame(list(paths_a.items())).set_index(0)
        paths_a.loc[nx.get_node_attributes(G,'pos').keys()] = t.max_depth+1
        shells = [paths_a[(paths_a == i).values].index.values for i in range(paths_a.max().iloc[0]+1)]
        shells[-1][-2:] = shells[-1][-2:][::-1]
    #     pos = nx.circular_layout(G)
        pos = (1j**(a['pos']*4)) * a['depth_r']
        pos = zip(pos.values.real, pos.values.imag)
        pos = {k : v for k,v in enumerate(pos)}
        if spring == True :
            pos = nx.spring_layout(G,.005,pos=pos,fixed=[0]+p['index'].tolist())
        edge_list = None
        if smooth_edges :
            edge_list = []
        options = {
            'node_color': nx.get_node_attributes(G,'color').values(),
            'edgelist' : edge_list,
            'pos': pos,
            'edge_color': nx.get_edge_attributes(G,'color').values(),
            'linewidths': 1,
            'width': 1,
    #         'node_size': nx.get_node_attributes(G,'node_size'),
    #         'line_color': 'grey',
    #         'label': labels,
    #         'label' : ['dead','alive'],
        }
        if node_size != None :
            options['node_size'] = node_size
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if smooth_edges :
            fig, ax = smoth_paths(G, fig, ax, pos, leaf_nodes, l_width=l_width, l_alpha=l_alpha, res=smooth_res, d=smooth_d)
        nx.draw(G, **options, node_shape='.')
        if draw_labels :
            labels = nx.get_edge_attributes(G,'label')
            labs = nx.draw_networkx_edge_labels(G,pos,edge_labels=labels, font_size=8,font_color='w',bbox=bbox)
        else :
            labels = None
            labs = None
        fig.set_facecolor(bg_color)
        score = round(clf.score(x,y), 4)
        n_row_samples = len(x)
        n_features = len(x.columns)
        file_name_parameters = (n_features,
                                int(score*10000),
                                '0' if n_row_samples is None else n_row_samples,
                                int(spring),
                                int(smooth_edges),
                                str(smooth_d),
                                str(l_alpha).split('.')[-1],
                                str(l_width).split('.')[-1],
                                img_res
                           )
        title = plt.title(f'{n_features} variables,  depth: {t.max_depth},  score: {score}', color='w')
        title_pos = title.get_position()
        cols = x.columns
        str_feat = '   '.join(x.columns.tolist())
        if np.char.isnumeric(cols.values.astype(str)).all() :
            cols = cols.astype(int)
            if sorted(cols) == list(range(min(cols), max(cols)+1)) :
                str_feat = edges_labels
                if edges_labels is None :
                    str_feat = f'range({x.columns[0]}, {x.columns[-1]})'
                    str_feat = ' '
        suptitle = plt.figtext(0.5, title_pos[1]*.88 , str_feat, color='w', horizontalalignment='center')
        file_name = './plots/nfeat{}_s{}_n{}_sp{}_sm{}_d{}_lw{}{}_dpi{}.png'.format(*file_name_parameters)
        if save_img :
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            plt.savefig(file_name, dpi=img_res, transparent=png_transparent)

    def smoth_paths(G, fig, ax, pos, leaf_nodes, l_width=1, l_alpha=1, res=50, d=3) :
        node_a, node_b = np.meshgrid(leaf_nodes,leaf_nodes)
        node_a = sum([r[i+1:].tolist() for i, r in enumerate(node_a)], [])
        node_b = sum([r[i+1:].tolist() for i, r in enumerate(node_b)], [])
        all_paths = [nx.shortest_path(G,a,b) for a, b in zip(node_a,node_b)]
        all_paths_pos = []
        for i, path in enumerate(tqdm(all_paths)) :
            path_pos = [pos[i] for i in path]
            all_paths_pos.append(path_pos)
            x, y = bspline(path_pos,n=res,degree=d,periodic=False).T
            c_first_node = nx.get_node_attributes(G,'color')[path[0]]
            c_last_node = nx.get_node_attributes(G,'color')[path[-1]]
            l_first = nx.get_node_attributes(G,'loss')[path[0]]
            l_last = nx.get_node_attributes(G,'loss')[path[-1]]
            cmap = LinearSegmentedColormap.from_list("reye", [c_first_node, c_last_node])
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm=plt.Normalize(0, 2)
            lc = LineCollection(segments, cmap=cmap)
            lc.set_array(np.arange(len(x)))
            lc.set_linewidth(l_width)
            lc.set_alpha(l_alpha)
            line = ax.add_collection(lc)
            line.set_zorder(-i-5000*(not bool(l_first))-5000*(not bool(l_first)))
        return fig, ax

    import scipy.interpolate as si
    def bspline(cv, n=50, degree=3, periodic=False):
        """ Calculate n samples on a bspline

            cv :      Array ov control vertices
            n  :      Number of samples to return
            degree:   Curve degree
            periodic: True - Curve is closed
        """
        cv = np.asarray(cv)
        count = cv.shape[0]
        if periodic:
            kv = np.arange(-degree,count+degree+1)
            factor, fraction = divmod(count+degree+1, count)
            cv = np.roll(np.concatenate((cv,) * factor + (cv[:fraction],)),-1,axis=0)
            degree = np.clip(degree,1,degree)
        else:
            degree = np.clip(degree,1,count-1)
            kv = np.clip(np.arange(count+degree+1)-degree,0,count-degree)
        max_param = count - (degree * (1-periodic))
        spl = si.BSpline(kv, cv, degree)
        return spl(np.linspace(0,max_param,n))


    if style=='radplot' :
        plt.style.use(['classic'])
        plt.rcParams['figure.figsize'] = (15, 15)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 14
        plt.rcParams['figure.dpi'] = fig_res
    if smooth_d is not None :
        smooth_edges=True
    if bbox == 'dark' :
        bbox = {'boxstyle':'round', 'ec':'k', 'fc':'k', 'alpha':0.8}
    if isinstance(X, np.ndarray) :
        cols = [str(c) for c in range(X.shape[-1]+1)]
        feature_cols = cols[:-1]
        label_col = cols[-1]
        data = pd.DataFrame([*X.T,Y], index=cols).T
    elif isinstance(X, pd.DataFrame) :
        if feature_cols is None :
            feature_cols = X.columns.tolist()
        if label_col is None :
            label_col = Y.columns.tolist()
        cols = feature_cols + label_col
        data = pd.concat([X, Y], axis=1)
    elif data is None :
        print('Data not provided. Please parse X and Y, or data arguments.')
    elif feature_cols is None :
        feature_cols = data.columns[:-1].tolist()
        label_col = data.columns[-1].values
    # if num_samples is None :
    #     num_samples = len(data)
    num_samples = min(num_samples, len(data))
    if levels is None :
        levels = data[feature_cols].astype('category').apply(lambda x: [x.cat.categories.tolist()])
    if not isinstance(feature_cols , list) :
        feature_cols = feature_cols.tolist()
    if not isinstance(label_col , list) :
        label_col = [label_col]
    x = data[feature_cols].sample(num_samples, random_state=random_state)
    y = data.loc[x.index,label_col]
    num_classes = int(y.max()[0] + 1)
    if cmap == 'pairs' :
        if num_classes <= 2 :
            cmap = ['#FF0000','#FFFF00','#0000FF','#00FFFF']
        if num_classes <= 3 :
            cmap = ['#FF0000','#FF00FF','#0000FF','#00FFFF','#00FF00','#FFFF00']
        else :
            cmap = [
                    '#880000','#FF4444',
                    '#880088','#FF44FF',
                    '#000088','#4444FF',
                    '#008888','#44FFFF',
                    '#008800','#44FF44',
                    '#888800','#FFFF44',
                    '#884400','#FFCC44',
                    '#880044','#FF44CC',
                    '#004488','#44CCFF',
                    '#440088','#CC44FF',
                    '#448800','#CCFF44',
                    '#008844','#44FFCC',
                    '#884488','#FFCCFF',
                    '#888844','#FFFFCC',
                    '#448888','#CCFFFF',
                   ]
# cmap = ['#880000','#FF0000','#880088','#FF00FF','#000088','#0000FF','#008888','#00FFFF','#008800','#00FF00','#888800','#FFFF00','#884400','#FF8800','#880044','#FF0088','#004488','#0088FF','#440088','#8800FF','#448800','#88FF00','#008844','#00FF88','#884488','#FF88FF','#888844','#FFFF88','#448888','#88FFFF',]
        cm = mpl.colors.ListedColormap(cmap,'labels_cmap',num_classes*2)
        cmap = [np.array(cm(i)[:3]) for i in range(num_classes*2)]
    elif isinstance(cmap, str) :
        cm = plt.get_cmap(cmap)
        cmap = [np.array(cm(i)[:3]) for i in range(num_classes*2)]
    label_hex_colors = np.array(cmap)

    G, node_paths, leaf_nodes = make_graph(x, y, clf, feature_cols, levels.copy(),
                                           tree_node_size, leaf_node_size, label_hex_colors=label_hex_colors,
                                           feat_short=edges_labels)

    df_depths, node_paths = get_depth(G, clf, node_paths)

    make_radial_plot(G, x, y, clf, df_depths, node_paths, leaf_nodes,
                     draw_labels=draw_labels, bbox=bbox, node_size=node_size,
                     l_width=l_width, l_alpha=l_alpha,
                     spring=spring, smooth_edges=smooth_edges,
                     smooth_d=smooth_d, smooth_res=smooth_res, edges_labels=edges_labels,
                     fig_res=72, save_img=save_img, img_res=img_res, png_transparent=png_transparent)


def quick_fitted_tree(X,Y, model_type=['GridSearch', 'FeatureSelection'], test_split=None, random_state=None) :
    from sklearn import tree, model_selection, feature_selection

    splitted_data = None
    sel_cols = None
    x = X.copy()
    y = Y.copy()
    if isinstance(test_split, (float)) :
        x, x_test, y, y_test = model_selection.train_test_split(x, y, test_size=test_split, random_state=random_state)
    cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = random_state )
    dtree = tree.DecisionTreeClassifier(random_state = random_state)
    model = dtree

    if 'FeatureSelection' in model_type :
        dtree_rfe = feature_selection.RFECV(tree.DecisionTreeClassifier(random_state=random_state), step = 1, scoring = 'accuracy', cv = cv_split)
        dtree_rfe.fit(x, y)
        x = x[:,dtree_rfe.get_support()]
        if isinstance(test_split, (float)) :
            x_test = x_test[:,dtree_rfe.get_support()]
        sel_cols = dtree_rfe.get_support()
    if 'GridSearch' in model_type :
        param_grid = {'criterion': ['gini', 'entropy'],
                      'max_depth': [2,4,6,8,10,None],
                      'random_state': [0],
                      #'splitter': ['best', 'random'],
                      #'min_samples_split': [2,5,10,.03,.05],
                      #'min_samples_leaf': [1,5,10,.03,.05],
                      #'max_features': [None, 'auto'],
                     }
        model = model_selection.GridSearchCV(tree.DecisionTreeClassifier(random_state=random_state), param_grid=param_grid, scoring = 'accuracy', cv = cv_split)

    model.fit(x, y)
    if model_type != None :
        model = model.best_estimator_

    if isinstance(test_split, (float)) :
        splitted_data = (x, x_test, y, y_test)
    else :
        splitted_data = (None, x, None, y)
    return model, sel_cols, splitted_data


def plot_pca(X,Y, validation_data=None, style='starplot', p_size=3.5, save_img=False, img_res=300, fig_res=72, random_state=None) :
    from sklearn.decomposition import PCA

    if validation_data is None :
        validation_data = (X,Y)
    X_valid, Y_valid = validation_data
    if style=='starplot' :
        plt.style.use(['dark_background'])
        plt.rcParams['figure.figsize'] = (15, 15)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 14
        plt.rcParams['figure.dpi'] = fig_res
    pca = PCA(2, random_state=random_state)
    pca.fit(X, Y)
    embedings = pca.transform(X_valid)
    embedings = np.array(embedings)
    size = p_size
    cmap = LinearSegmentedColormap.from_list("recy", ["magenta","cyan"])
    for point in range(1,10) :
        plt.scatter(embedings[:,0],embedings[:,1], c=Y_valid.ravel(), cmap=cmap, s=5*point**size, alpha=1/(point**size), edgecolors='', )
    file_name = './plots/s' + str(int(size)) + '_pca.png'
    if save_img :
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name, dpi=img_res, transparent=True)
    plt.show()

def plot_umap(X,Y, validation_data=None, style='starplot', p_size=3.5, save_img=False, img_res=300, fig_res=72, random_state=None) :
    from umap import UMAP

    if validation_data is None :
        validation_data = (X,Y)
    X_valid, Y_valid = validation_data
    if style=='starplot' :
        plt.style.use(['dark_background'])
        plt.rcParams['figure.figsize'] = (15, 15)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.size'] = 14
        plt.rcParams['figure.dpi'] = fig_res
    umap = UMAP(25, random_state=random_state)
    umap.fit(X, Y.ravel())
    embedings = umap.transform(X_valid)
    embedings = np.array(embedings)
    size = p_size
    cmap = LinearSegmentedColormap.from_list("recy", ["magenta","cyan"])
    for point in range(1,10) :
        plt.scatter(embedings[:,0],embedings[:,1], c=Y_valid.ravel(), cmap=cmap, s=5*point**size, alpha=1/(point**size), edgecolors='', )
    file_name = './plots/s' + str(int(size)) + '_umap.png'
    if save_img :
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        plt.savefig(file_name, dpi=img_res, transparent=True)
    plt.show()

def plot_tsne(X,Y, style='starplot', p_size=3.5, save_img=False, img_res=300, fig_res=72, random_state=None) :
        from sklearn.manifold import TSNE

        if style=='starplot' :
            plt.style.use(['dark_background'])
            plt.rcParams['figure.figsize'] = (15, 15)
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.size'] = 14
            plt.rcParams['figure.dpi'] = fig_res
        tsne = TSNE(2, random_state=random_state, perplexity=40)
        embedings = tsne.fit_transform(X)
        embedings = np.array(embedings)
        size = p_size
        cmap = LinearSegmentedColormap.from_list("recy", ["magenta","cyan"])
        for point in range(1,10) :
            plt.scatter(embedings[:,0],embedings[:,1], c=Y.ravel(), cmap=cmap, s=5*point**size, alpha=1/(point**size), edgecolors='', )
        file_name = './plots/s' + str(int(size)) + '_tsne.png'
        if save_img :
            os.makedirs(os.path.dirname(file_name), exist_ok=True)
            plt.savefig(file_name, dpi=img_res, transparent=True)
        plt.show()
