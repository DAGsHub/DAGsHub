import enum
from abc import ABCMeta
from typing import Set, Optional


class ReservedTags(enum.Enum):
    """:meta private:"""

    ANNOTATION = "annotation"
    DOCUMENT = "text_document"
    THUMBNAIL_VIZ = "thumbnail_viz"
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"


# These are the base primitives that the data engine database is capable of storing
class MetadataFieldType(enum.Enum):
    """
    Backing types in the Data Engine's database
    """

    BOOLEAN = "BOOLEAN"
    """Python's ``bool``"""
    INTEGER = "INTEGER"
    """Python's ``int``"""
    DATETIME = "DATETIME"
    """Python's ``datetime.datetime``"""
    FLOAT = "FLOAT"
    """Python's ``float``"""
    STRING = "STRING"
    """Python's ``str``"""
    BLOB = "BLOB"
    """Python's ``bytes``"""


# this extends MetadataFieldType, it appears as data type in schema but only for querying
# i.e no field is actually of this type
DATETIME_RANGE = "DATETIME_RANGE"


class ThumbnailType(enum.Enum):
    """
    Thumbnail types for visualization

    :meta private:
    """

    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    PDF = "pdf"
    TEXT = "text"


class DagshubDataType(metaclass=ABCMeta):
    """
    Inheritors of this ABC define custom types

    They are backed by a primitive type, but they also may have additional tags, which we use to enhance the experience.

    Attributes:
        backing_field_type: primitive type in the data engine database
        custom_tags: additional tags applied to this type
    """

    backing_field_type: Optional[MetadataFieldType] = None
    custom_tags: Optional[Set[str]] = None


class Int(DagshubDataType):
    """Basic python ``int``"""

    backing_field_type = MetadataFieldType.INTEGER


class DateTime(DagshubDataType):
    """Basic python ``datetime.datetime``

    .. note::
        dagshub backend can receive only an integer timestamp (utc timestamp).
        in the below example the dagshub client sends int(t.timestamp()) to the backend
        if you want to save your own timestamp it must be rounded like this.

    Example::

        datapoints = datasource.all()
        t = dateutil.parser.parse("2022-04-05T15:30:00.99999+05:30")
        datapoints[path][name] = t
        datapoints[path].save()
    """
    backing_field_type = MetadataFieldType.DATETIME


class String(DagshubDataType):
    """Basic python ``str``"""

    backing_field_type = MetadataFieldType.STRING


class Blob(DagshubDataType):
    """
    Basic python ``bytes``

    .. note::
        DagsHub doesn't return the blob fields by default, instead returning their hashes.
        Check out :func:`Datapoint.get_blob() <dagshub.data_engine.model.datapoint.Datapoint.get_blob>`
        to learn how to download the blob value.
    """

    backing_field_type = MetadataFieldType.BLOB


class Float(DagshubDataType):
    """
    Basic python ``float``
    """

    backing_field_type = MetadataFieldType.FLOAT


class Bool(DagshubDataType):
    """
    Basic python ``bool``
    """

    backing_field_type = MetadataFieldType.BOOLEAN


class LabelStudioAnnotation(DagshubDataType):
    """
    LabelStudio annotation. Backing type is blob.
    Has the annotation tag set.
    """

    backing_field_type = MetadataFieldType.BLOB
    custom_tags = {ReservedTags.ANNOTATION.value}


class Voxel51Annotation(DagshubDataType):
    """
    Voxel51 annotation. Backing type is blob.
    Has the annotation tag set.
    """

    backing_field_type = MetadataFieldType.BLOB
    custom_tags = {ReservedTags.ANNOTATION.value}


class Document(DagshubDataType):
    """
    Field with large text values that is stored as a blob.
    Document fields can't be filtered on,
    but allow you to store arbitrarily large text longer than allowed 512 characters
    """

    backing_field_type = MetadataFieldType.BLOB
    custom_tags = {ReservedTags.DOCUMENT.value}
