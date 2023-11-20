import enum
import logging
from typing import TYPE_CHECKING, Optional, Union, Dict

from treelib import Tree, Node

from dagshub.data_engine.model.errors import WrongOperatorError
from dagshub.data_engine.model.schema_util import metadataTypeLookup, metadataTypeLookupReverse

if TYPE_CHECKING:
    from dagshub.data_engine.model.datasource import Datasource

logger = logging.getLogger(__name__)


def bytes_deserializer(val: str) -> bytes:
    if val.startswith('b"') or val.startswith("b'"):
        return val[2:-1].encode()
    # Fallback - encode whatever we got from the server
    return val.encode()


_metadataTypeCustomConverters = {
    bool: lambda x: x.lower() == "true",
    bytes: bytes_deserializer,
}


class FieldFilterOperand(enum.Enum):
    EQUAL = "EQUAL"
    GREATER_THAN = "GREATER_THAN"
    GREATER_EQUAL_THAN = "GREATER_EQUAL_THAN"
    LESS_THAN = "LESS_THAN"
    LESS_EQUAL_THAN = "LESS_EQUAL_THAN"
    CONTAINS = "CONTAINS"
    IS_NULL = "IS_NULL"


fieldFilterOperandMap = {
    "eq": FieldFilterOperand.EQUAL,
    "gt": FieldFilterOperand.GREATER_THAN,
    "ge": FieldFilterOperand.GREATER_EQUAL_THAN,
    "lt": FieldFilterOperand.LESS_THAN,
    "le": FieldFilterOperand.LESS_EQUAL_THAN,
    "contains": FieldFilterOperand.CONTAINS,
    "isnull": FieldFilterOperand.IS_NULL,
}

fieldFilterOperandMapReverseMap: Dict[str, str] = {}
for k, v in fieldFilterOperandMap.items():
    fieldFilterOperandMapReverseMap[v.value] = k

UNFILLED_NODE_TAG = "undefined"


class DatasourceQuery:
    def __init__(self, column_or_query: Optional[Union[str, "DatasourceQuery"]] = None, as_of: Optional[int] = None):
        self._operand_tree: Tree = Tree()
        if type(column_or_query) is str:
            # If it's ds["column"] then the root node is just the column name, will be filled later
            data = {"field": column_or_query}
            if as_of:
                data["as_of"] = int(as_of)
            self._operand_tree.create_node(UNFILLED_NODE_TAG, data=data)
        elif column_or_query is not None:
            self._operand_tree.create_node(column_or_query)

        if as_of:
            self._as_of = as_of

    def __repr__(self):
        if self.is_empty:
            return "Query: empty"
        return f"Query: {self.to_dict()}"

    @property
    def column_filter(self) -> Optional[str]:
        filter_node = self._column_filter_node
        if filter_node is None:
            return None
        return filter_node.data["field"]

    def compose(self, op: str, other: Optional[Union[str, int, float, "DatasourceQuery", "Datasource"]]):
        if self._column_filter_node is not None:
            # If there was an unfilled query node with a column - put the operand in that node
            node = self._column_filter_node
            node.tag = op
            node.data.update({"value": other})
        elif op == "isnull":
            # Can only do isnull on the column filter, if we got here, there's something wrong
            raise RuntimeError("is_null operation can only be done on a column (e.g. ds['col1'].is_null())")
        elif op == "not":
            new_tree = Tree()
            not_node = new_tree.create_node("not")
            new_tree.paste(not_node.identifier, self._operand_tree)
            self._operand_tree = new_tree
        else:
            # The query is an actual query with a tree - make a subtree
            if type(other) is not DatasourceQuery:
                raise RuntimeError(f"Expected other argument to be a dataset, got {type(other)} instead")
            if op not in ["and", "or"]:
                raise RuntimeError(
                    f"Cannot use operator '{op}' to chain two queries together.\r\n"
                    f"Queries:\r\n"
                    f"\t{self}\r\n"
                    f"\t{other}\r\n"
                )
            # Don't compose with an empty query, carry the other instead
            if self.is_empty:
                self._operand_tree = other._operand_tree
                return
            elif other.is_empty and other._column_filter_node is None:
                return
            composite_tree = Tree()
            root_node = composite_tree.create_node(op)
            composite_tree.paste(root_node.identifier, self._operand_tree)
            composite_tree.paste(root_node.identifier, other._operand_tree)
            self._operand_tree = composite_tree

    @property
    def _column_filter_node(self) -> Node:
        return next(self._operand_tree.filter_nodes(lambda n: n.tag == UNFILLED_NODE_TAG), None)

    @property
    def _operand_root(self) -> Node:
        return self._operand_tree[self._operand_tree.root]

    def serialize_graphql(self):
        if self.is_empty:
            return None
        return self._serialize_node(self._operand_root, self._operand_tree)

    @staticmethod
    def _serialize_node(node: Node, tree: Tree) -> Dict:
        operand = node.tag
        if operand in ["and", "or"]:
            # recursively serialize children subqueries
            return {operand: [DatasourceQuery._serialize_node(child, tree) for child in tree.children(node.identifier)]}
        if operand == "not":
            assert len(tree.children(node.identifier)) == 1
            child = tree.children(node.identifier)[0]
            serialized = DatasourceQuery._serialize_node(child, tree)
            serialized["not"] = True
            return serialized
        else:
            query_op = fieldFilterOperandMap.get(operand)
            if query_op is None:
                raise WrongOperatorError(f"Operator {operand} is not supported")
            key = node.data["field"]
            value = node.data["value"]
            as_of = node.data.get("as_of")

            value_type = metadataTypeLookup[type(value)].value
            if type(value) is bytes:
                # TODO: this will need to probably be changed when we allow actual binary field comparisons
                value = value.decode("utf-8")
            else:
                value = str(value)
            if value_type is None:
                raise RuntimeError(
                    f"Value type {value_type} is not supported for querying.\r\n"
                    f"Supported types: {list(metadataTypeLookup.keys())}"
                )
            res = {
                "filter": {
                    "key": key,
                    "value": str(value),
                    "valueType": value_type,
                    "comparator": query_op.value
                }
            }
            if as_of:
                res["filter"]["asOf"] = as_of
            return res

    @staticmethod
    def deserialize(serialized_query: Dict) -> "DatasourceQuery":
        q = DatasourceQuery()
        op_tree = Tree()

        DatasourceQuery._deserialize_node(serialized_query, op_tree)

        q._operand_tree = op_tree
        return q

    @staticmethod
    def _deserialize_node(node_dict: Dict, tree: Tree, parent_node=None) -> None:
        keys = list(node_dict.keys())

        is_negative = node_dict.get("not", False)
        if is_negative:
            # If operation is negative - prepend a "not" node to the node we'll be adding
            neg_node = Node(tag="not")
            tree.add_node(neg_node, parent_node)
            parent_node = neg_node

        op_type = keys[0]
        val = node_dict[op_type]
        # Types: and, or, filter
        if op_type == "filter":
            comparator = fieldFilterOperandMapReverseMap[val["comparator"]]
            key = val["key"]
            value_type = metadataTypeLookupReverse[val["valueType"]]
            converter = _metadataTypeCustomConverters.get(value_type, lambda x: value_type(x))
            value = converter(val["value"])
            as_of = val.get("asOf")
            node = Node(tag=comparator, data={"field": key, "value": value, "as_of": as_of})
            tree.add_node(node, parent_node)
        elif op_type in ("and", "or"):
            main_node = Node(tag=op_type)
            tree.add_node(main_node, parent_node)
            for nested_node in val:
                DatasourceQuery._deserialize_node(nested_node, tree, main_node)
        else:
            raise RuntimeError(f"Unknown serialized query dict: {node_dict}")

    def to_dict(self):
        return self._operand_tree.to_dict(with_data=True)

    def __deepcopy__(self, memodict={}):
        q = DatasourceQuery()
        q._operand_tree = Tree(tree=self._operand_tree, deep=True)
        return q

    @property
    def is_empty(self):
        return self._operand_tree.root is None or self._column_filter_node is not None
