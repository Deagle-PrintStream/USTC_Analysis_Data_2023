__doc__ = """ 
A simple script for association rules finding by FP growth algorithm.
"""

__all__ = ["association_rules"]
__author__ = "suiyi_liu@mail.ustc.edu.cn"
__version__ = "1.0"

import os, sys
import pandas as pd
from tqdm import tqdm


global_consquent:str=""

def save_rule(rule, path):
    """Save the rule list into a `.txt` file"""
    with open(path, "w") as f:
        f.write("index  confidence" + "   rules\n")
        index = 1
        for item in rule:
            s = " {:<4d}  {:.3f}        {}=>{}\n".format(
                index, item[2], str(list(item[0])), str(list(item[1]))
            )
            index += 1
            f.write(s)
        f.close()
    print("result saved in {}".format(path))


class Node:
    """class for node in FP tree"""


    def __init__(self, node_name, count, parentNode):
        self.name = node_name
        self.count = count
        self.nodeLink: Node | None = None
        """whereby which all nodes with same node_name can be found in the whole tree"""

        self.parent = parentNode
        self.children: dict = {}
        """child node container with struct of {node_name:node_addr}"""


class Fp_growth:
    """A simple implementation of FP growth algorithm"""

    @staticmethod
    def update_header(node: Node, targetNode):
        """update the node chain in headertable"""
        while node.nodeLink != None:
            node = node.nodeLink
        node.nodeLink = targetNode

    @staticmethod
    def update_fptree(items, node: Node, headerTable: dict) -> None:
        if items[0] in node.children:
            node.children[items[0]].count += 1
        else:
            node.children[items[0]] = Node(items[0], 1, node)

            if headerTable[items[0]][1] == None:
                headerTable[items[0]][1] = node.children[items[0]]
            else:
                Fp_growth.update_header(
                    headerTable[items[0]][1], node.children[items[0]]
                )

        if len(items) > 1:
            Fp_growth.update_fptree(items[1:], node.children[items[0]], headerTable)

    @staticmethod
    def create_fptree(
        data_set: list[list], min_support: float, flag=False
    ) -> dict | None:
        """
        main function to create the FP tree,
        sturct of header_table is {"nodename":[num,node],..}
        """
        #threshold for support
        consquent_count=0
        for record in data_set:
            if global_consquent in record:
                consquent_count+=1
        
        item_count = {}  # count occurence of all combinations
        for t in data_set:
            for item in t:
                if item not in item_count:
                    item_count[item] = 1
                else:
                    item_count[item] += 1

        headerTable = {}
        # WARING : convert min_support from float type into int
        #-----------------------------------------------------------------------------------
        threshold = min_support * consquent_count
        for k in item_count:  # kick out all combinations with unsatisfied confidence
            if item_count[k] >= threshold:
                headerTable[k] = item_count[k]

        freqItemSet = set(headerTable.keys())  # combination satisfied
        if len(freqItemSet) == 0:
            return None
        for k in headerTable:
            headerTable[k] = [headerTable[k], None]  # element: [count, node]
        tree_header = Node("head node", 1, None)
        if flag:
            ite = tqdm(data_set)
        else:
            ite = data_set

        for t in ite:
            localD = {}
            for item in t:
                if item in freqItemSet:  # filteration for high confidence
                    localD[item] = headerTable[item][0]  # element : count
            if len(localD) > 0:
                # sort all samples with global frequence
                order_item = [
                    v[0]
                    for v in sorted(localD.items(), key=lambda x: x[1], reverse=True)
                ]
                # build up tree with filtered data
                Fp_growth.update_fptree(order_item, tree_header, headerTable)
        return headerTable

    @staticmethod
    def find_path(node: Node, nodepath: list) -> None:
        """
        add all parentNode into path
        """
        if node.parent != None:
            nodepath.append(node.parent.name)
            Fp_growth.find_path(node.parent, nodepath)

    @staticmethod
    def find_cond_pattern_base(node_name, headerTable) -> dict:
        """
        find all basement of conditions
        """
        treeNode = headerTable[node_name][1]
        cond_pat_base = {}
        while treeNode != None:
            nodepath = []
            Fp_growth.find_path(treeNode, nodepath)
            if len(nodepath) > 1:
                cond_pat_base[frozenset(nodepath[:-1])] = treeNode.count
            treeNode = treeNode.nodeLink
        return cond_pat_base

    @staticmethod
    def create_cond_fptree(
        headerTable, min_support: float, temp: set, freq_items: set, support_data: dict
    ):
        freqs = [
            v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])
        ]  # sort by frequence
        for freq in freqs:
            freq_set = temp.copy()
            freq_set.add(freq)
            freq_items.add(frozenset(freq_set))
            if (
                frozenset(freq_set) not in support_data
            ):  # check if this combination in support_set
                support_data[frozenset(freq_set)] = headerTable[freq][0]
            else:
                support_data[frozenset(freq_set)] += headerTable[freq][0]

            cond_pat_base = Fp_growth.find_cond_pattern_base(
                freq, headerTable
            )  # find all basement of condition mode
            cond_pat_dataset = []  # turn into list
            for item in cond_pat_base:
                item_temp = list(item)
                item_temp.sort()
                for i in range(cond_pat_base[item]):
                    cond_pat_dataset.append(item_temp)
            # create condition mode tree
            cur_headtable = Fp_growth.create_fptree(cond_pat_dataset, min_support)
            if cur_headtable != None:
                Fp_growth.create_cond_fptree(
                    cur_headtable, min_support, freq_set, freq_items, support_data
                )

    @staticmethod
    def generate_L(data_set: list, min_support: float) -> tuple[list[set], dict]:
        freqItemSet = set()
        support_data = {}
        headerTable = Fp_growth.create_fptree(data_set, min_support, flag=True)
        # for each combination, create an individual FP tree and keep digging
        Fp_growth.create_cond_fptree(
            headerTable, min_support, set(), freqItemSet, support_data
        )

        max_l = 0
        for i in freqItemSet:  # save the list into container
            if len(i) > max_l:
                max_l = len(i)
        L = [set() for _ in range(max_l)]
        for i in freqItemSet:
            L[len(i) - 1].add(i)
        for i in range(len(L)):
            print("frequent item {}:{}".format(i + 1, len(L[i])))
        return L, support_data

    @staticmethod
    def generate_R(
        data_set: list, min_support: float, min_conf: float, consquent: str
    ) -> list[tuple[set, set, float]]:
        L, support_data = Fp_growth.generate_L(data_set, min_support)
        rule_list: list[tuple[set, set, float]] = []
        sub_set_list: list[set] = []
        for i in range(0, len(L)):
            for freq_set in L[i]:
                for sub_set in sub_set_list:
                    # filter the target feature as consquenct
                    #-----------------------------------------------------------------------
                    if (
                        consquent in sub_set and 
                        sub_set.issubset(freq_set) and
                        freq_set - sub_set in support_data
                    ):
                        # add freq_set-sub_set into support_data
                        conf = support_data[freq_set] / support_data[freq_set - sub_set]
                        big_rule: tuple[set, set, float] = (
                            freq_set - sub_set,
                            sub_set,
                            conf,
                        )
                        if conf >= min_conf and big_rule not in rule_list:
                            rule_list.append(big_rule)

                sub_set_list.append(freq_set)

        rule_list = sorted(rule_list, key=lambda x: (x[2]), reverse=True)
        return rule_list


def preprocess(df: pd.DataFrame) -> list[list[str]]:
    df = df.astype("str")
    record_list: list[list[str]] = []
    try:
        for i in range(0, len(df)):
            record = df.iloc[i].to_list()
            record.sort()
            record_list.append(record)  # type: ignore
    except TypeError:
        raise TypeError("invalid feature value")
    return record_list


def association_rules(
    df: pd.DataFrame,
    consquent: str = "REPEAT",
    min_support: float = 0.6,
    min_conf: float = 0.15,
    save_path: str = "./rules.txt",
) -> list[tuple] | None:
    os.chdir(sys.path[0])
    global global_consquent
    global_consquent=consquent
    record_list: list = preprocess(df)
    fp = Fp_growth()

    rule_list = fp.generate_R(record_list, min_support, min_conf, consquent)
    if type(save_path) == str:
        save_rule(rule_list, save_path)
    return rule_list


def main() -> None:
    # association_rules()
    pass


if __name__ == "__main__":
    main()
