from sklearn.tree.export import _DOTTreeExporter
from sklearn.tree  import _tree
from sklearn.ensemble import IsolationForest
from io import StringIO
import numpy as np


class _DOTIFTreeExporter(_DOTTreeExporter):
    def __init__(self, meta_data, out_file=None):
        super().__init__(out_file=out_file,
                         filled=True)

        self.meta_data = meta_data

    def export(self, decision_tree):
        self.head()
        self.recurse(decision_tree.tree_, 0, 
                     criterion=decision_tree.criterion)
        self.tail()

    def recurse(self, tree, node_id, criterion, parent=None, depth=0):
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        self.out_file.write(
            '%d [label=%s' % (node_id, self.node_to_str(tree, node_id,
                                                        criterion)))

        self.out_file.write(', fillcolor="%s"'
                               % self.get_fill_color(tree, node_id))

        self.out_file.write('] ;\n')

        if parent is not None:
            self.out_file.write('%d -> %d' % (parent, node_id))

            if parent == 0:
                angles = np.array([45, -45]) * ((self.rotate - .5) * -2)
                self.out_file.write(' [labeldistance=2.5, labelangle=')
                 
                if node_id == 1:
                    self.out_file.write('%d, headlabel="True"]' %
                                            angles[0])
                else:
                    self.out_file.write('%d, headlabel="False"]' %
                                            angles[1])
            self.out_file.write(' ;\n')

        if left_child != _tree.TREE_LEAF:
            self.recurse(tree, left_child, criterion=criterion,
                         parent=node_id, depth=depth + 1)
            self.recurse(tree, right_child, criterion=criterion,
                         parent=node_id, depth=depth + 1)


    def get_fill_color(self, tree, node_id):
        code = '#C0C0C0'
       
        count0 = self.meta_data[node_id][0]
        count1 = self.meta_data[node_id][1]
        prediction = self.meta_data[node_id][2]

        if count0 > count1:
            code = '#ccffcc' 
        else:
            code = '#ffcccc' 

        if prediction == 1:
            code = '#009900' 
        elif prediction == -1:
            code = '#990000' 

        return code


def get_meta_data(model, data, predictions):

    meta_data_node = {}
    meta_data_path = {}

    for tree in model.estimators_:
        leaves_index = tree.apply(data)
        node_indicator = tree.decision_path(data)

        for prediction, index, path in zip(predictions, 
                                           leaves_index, 
                                           node_indicator):

            # meta_data_node            
            for ii, value in enumerate(path.toarray()[0]):
                if ii > index:
                    continue

                if value == 0:
                    continue

                if not ii in meta_data_node.keys():
                    # [normal count, anomaly count, prediction]
                    meta_data_node[ii] = [0, 0, 0] 
                  
                if prediction == 1:
                    meta_data_node[ii][0] += 1
                else:
                    meta_data_node[ii][1] += 1

                if ii == index:
                    meta_data_node[ii][2] = prediction    

            # meta_data_path
            if prediction == 1:
                continue

            paths = path.toarray()[0]
            thresholds = []
            features = []
            directions = []
            parent = -999

            for ii in range(0, index+1):

                if paths[ii] == 0:
                    continue
                 
                thresholds.append(tree.tree_.threshold[ii])
                features.append(tree.tree_.feature[ii])

                if (parent+1) == ii:
                    directions.append(True)
                else:
                    directions.append(False)
                parent = ii

            if not index in meta_data_path.keys():
                meta_data_path[index] = [thresholds, 
                                         features,
                                         directions]   
            
        break
  
    return meta_data_node, meta_data_path


def export_if_graphviz(model, data, predictions):

    # get meta information
    meta_data, dummy = get_meta_data(model, data, predictions)

    # graphviz
    out_file = StringIO()
    exporter = _DOTIFTreeExporter(meta_data, out_file=out_file)
    exporter.export(model.estimators_[0])

    return exporter.out_file.getvalue()

def export_if_text(model, data, predictions):
    
    # get meta information
    dummy, meta_data = get_meta_data(model, data, predictions)
    
    print ('inf> features and thresholds for anomalies')
    for key, values in meta_data.items():
        print ('inf> +-- node index %s' % key)
        
        for threshold, feature, direction in zip(values[0][:-1], 
                                      values[1][:-1],
                                      values[2][1:]):
            arrow = '<='
            if direction == False:
                arrow = '> '
            print ('inf>  +-- feature %s %s %s' % (feature, 
                                                 arrow,  
                                                 threshold))
         
    
