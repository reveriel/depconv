# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: second/protos/train.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from second.protos import optimizer_pb2 as second_dot_protos_dot_optimizer__pb2
from second.protos import preprocess_pb2 as second_dot_protos_dot_preprocess__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='second/protos/train.proto',
  package='second.protos',
  syntax='proto3',
  serialized_options=None,
  serialized_pb=_b('\n\x19second/protos/train.proto\x12\rsecond.protos\x1a\x1dsecond/protos/optimizer.proto\x1a\x1esecond/protos/preprocess.proto\"\xfa\x01\n\x0bTrainConfig\x12+\n\toptimizer\x18\x01 \x01(\x0b\x32\x18.second.protos.Optimizer\x12\r\n\x05steps\x18\x02 \x01(\r\x12\x16\n\x0esteps_per_eval\x18\x03 \x01(\r\x12\x1d\n\x15save_checkpoints_secs\x18\x04 \x01(\r\x12\x1a\n\x12save_summary_steps\x18\x05 \x01(\r\x12\x1e\n\x16\x65nable_mixed_precision\x18\x06 \x01(\x08\x12\x19\n\x11loss_scale_factor\x18\x07 \x01(\x02\x12!\n\x19\x63lear_metrics_every_epoch\x18\x08 \x01(\x08\x62\x06proto3')
  ,
  dependencies=[second_dot_protos_dot_optimizer__pb2.DESCRIPTOR,second_dot_protos_dot_preprocess__pb2.DESCRIPTOR,])




_TRAINCONFIG = _descriptor.Descriptor(
  name='TrainConfig',
  full_name='second.protos.TrainConfig',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='optimizer', full_name='second.protos.TrainConfig.optimizer', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='steps', full_name='second.protos.TrainConfig.steps', index=1,
      number=2, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='steps_per_eval', full_name='second.protos.TrainConfig.steps_per_eval', index=2,
      number=3, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='save_checkpoints_secs', full_name='second.protos.TrainConfig.save_checkpoints_secs', index=3,
      number=4, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='save_summary_steps', full_name='second.protos.TrainConfig.save_summary_steps', index=4,
      number=5, type=13, cpp_type=3, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='enable_mixed_precision', full_name='second.protos.TrainConfig.enable_mixed_precision', index=5,
      number=6, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='loss_scale_factor', full_name='second.protos.TrainConfig.loss_scale_factor', index=6,
      number=7, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='clear_metrics_every_epoch', full_name='second.protos.TrainConfig.clear_metrics_every_epoch', index=7,
      number=8, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=108,
  serialized_end=358,
)

_TRAINCONFIG.fields_by_name['optimizer'].message_type = second_dot_protos_dot_optimizer__pb2._OPTIMIZER
DESCRIPTOR.message_types_by_name['TrainConfig'] = _TRAINCONFIG
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

TrainConfig = _reflection.GeneratedProtocolMessageType('TrainConfig', (_message.Message,), dict(
  DESCRIPTOR = _TRAINCONFIG,
  __module__ = 'second.protos.train_pb2'
  # @@protoc_insertion_point(class_scope:second.protos.TrainConfig)
  ))
_sym_db.RegisterMessage(TrainConfig)


# @@protoc_insertion_point(module_scope)
