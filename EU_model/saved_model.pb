юР$
ЏВ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	ђљ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ѕ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
Г
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(ѕ
?
Mul
x"T
y"T
z"T"
Ttype:
2	љ

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
Ї
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0ѕ
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
┴
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕе
@
StaticRegexFullMatch	
input

output
"
patternstring
Ш
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
░
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
Ъ
TensorListReserve
element_shape"
shape_type
num_elements(
handleіжУelement_dtype"
element_dtypetype"

shape_typetype:
2	
ѕ
TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsint         
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
ќ
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ѕ
ћ
While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
ѕ"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68ђ┐#
z
dense_66/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_66/kernel
s
#dense_66/kernel/Read/ReadVariableOpReadVariableOpdense_66/kernel*
_output_shapes

:*
dtype0
r
dense_66/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_66/bias
k
!dense_66/bias/Read/ReadVariableOpReadVariableOpdense_66/bias*
_output_shapes
:*
dtype0
z
dense_67/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_67/kernel
s
#dense_67/kernel/Read/ReadVariableOpReadVariableOpdense_67/kernel*
_output_shapes

:*
dtype0
r
dense_67/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_67/bias
k
!dense_67/bias/Read/ReadVariableOpReadVariableOpdense_67/bias*
_output_shapes
:*
dtype0
z
dense_68/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_68/kernel
s
#dense_68/kernel/Read/ReadVariableOpReadVariableOpdense_68/kernel*
_output_shapes

:*
dtype0
r
dense_68/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_68/bias
k
!dense_68/bias/Read/ReadVariableOpReadVariableOpdense_68/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
њ
lstm_22/lstm_cell_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namelstm_22/lstm_cell_22/kernel
І
/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/kernel*
_output_shapes

:*
dtype0
д
%lstm_22/lstm_cell_22/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%lstm_22/lstm_cell_22/recurrent_kernel
Ъ
9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_22/lstm_cell_22/recurrent_kernel*
_output_shapes

:*
dtype0
і
lstm_22/lstm_cell_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_22/lstm_cell_22/bias
Ѓ
-lstm_22/lstm_cell_22/bias/Read/ReadVariableOpReadVariableOplstm_22/lstm_cell_22/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
ѕ
Adam/dense_66/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_66/kernel/m
Ђ
*Adam/dense_66/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_66/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_66/bias/m
y
(Adam/dense_66/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_67/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_67/kernel/m
Ђ
*Adam/dense_67/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_67/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_67/bias/m
y
(Adam/dense_67/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_68/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_68/kernel/m
Ђ
*Adam/dense_68/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/m*
_output_shapes

:*
dtype0
ђ
Adam/dense_68/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_68/bias/m
y
(Adam/dense_68/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/m*
_output_shapes
:*
dtype0
а
"Adam/lstm_22/lstm_cell_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/lstm_22/lstm_cell_22/kernel/m
Ў
6Adam/lstm_22/lstm_cell_22/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_22/kernel/m*
_output_shapes

:*
dtype0
┤
,Adam/lstm_22/lstm_cell_22/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m
Г
@Adam/lstm_22/lstm_cell_22/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m*
_output_shapes

:*
dtype0
ў
 Adam/lstm_22/lstm_cell_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_22/lstm_cell_22/bias/m
Љ
4Adam/lstm_22/lstm_cell_22/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_22/bias/m*
_output_shapes
:*
dtype0
ѕ
Adam/dense_66/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_66/kernel/v
Ђ
*Adam/dense_66/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_66/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_66/bias/v
y
(Adam/dense_66/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_66/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_67/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_67/kernel/v
Ђ
*Adam/dense_67/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_67/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_67/bias/v
y
(Adam/dense_67/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_67/bias/v*
_output_shapes
:*
dtype0
ѕ
Adam/dense_68/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_68/kernel/v
Ђ
*Adam/dense_68/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/kernel/v*
_output_shapes

:*
dtype0
ђ
Adam/dense_68/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_68/bias/v
y
(Adam/dense_68/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_68/bias/v*
_output_shapes
:*
dtype0
а
"Adam/lstm_22/lstm_cell_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/lstm_22/lstm_cell_22/kernel/v
Ў
6Adam/lstm_22/lstm_cell_22/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_22/lstm_cell_22/kernel/v*
_output_shapes

:*
dtype0
┤
,Adam/lstm_22/lstm_cell_22/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v
Г
@Adam/lstm_22/lstm_cell_22/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v*
_output_shapes

:*
dtype0
ў
 Adam/lstm_22/lstm_cell_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_22/lstm_cell_22/bias/v
Љ
4Adam/lstm_22/lstm_cell_22/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_22/lstm_cell_22/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
Э?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*│?
valueЕ?Bд? BЪ?
У
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
д

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
┴
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses*
д

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
д

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
Т
/iter

0beta_1

1beta_2
	2decay
3learning_ratemjmkml mm'mn(mo4mp5mq6mrvsvtvu vv'vw(vx4vy5vz6v{*
C
0
1
42
53
64
5
 6
'7
(8*
C
0
1
42
53
64
5
 6
'7
(8*
* 
░
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

<serving_default* 
_Y
VARIABLE_VALUEdense_66/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_66/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 
Њ
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
с
B
state_size

4kernel
5recurrent_kernel
6bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G_random_generator
H__call__
*I&call_and_return_all_conditional_losses*
* 

40
51
62*

40
51
62*
* 
Ъ

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 
_Y
VARIABLE_VALUEdense_67/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_67/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 
Њ
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*
* 
* 
_Y
VARIABLE_VALUEdense_68/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_68/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 
Њ
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUElstm_22/lstm_cell_22/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_22/lstm_cell_22/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_22/lstm_cell_22/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

Z0
[1*
* 
* 
* 
* 
* 
* 
* 
* 
* 

40
51
62*

40
51
62*
* 
Њ
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*
* 
* 
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	atotal
	bcount
c	variables
d	keras_api*
H
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api*
* 
* 
* 
* 
* 
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

a0
b1*

c	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

e0
f1*

h	variables*
ѓ|
VARIABLE_VALUEAdam/dense_66/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_66/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_67/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_67/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_68/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_68/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_22/lstm_cell_22/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/lstm_22/lstm_cell_22/recurrent_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_22/lstm_cell_22/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_66/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_66/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_67/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_67/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ѓ|
VARIABLE_VALUEAdam/dense_68/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_68/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_22/lstm_cell_22/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѕѓ
VARIABLE_VALUE,Adam/lstm_22/lstm_cell_22/recurrent_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_22/lstm_cell_22/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Ѕ
serving_default_dense_66_inputPlaceholder*+
_output_shapes
:         *
dtype0* 
shape:         
і
StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_66_inputdense_66/kerneldense_66/biaslstm_22/lstm_cell_22/kernellstm_22/lstm_cell_22/bias%lstm_22/lstm_cell_22/recurrent_kerneldense_67/kerneldense_67/biasdense_68/kerneldense_68/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8ѓ *-
f(R&
$__inference_signature_wrapper_442918
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
»
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_66/kernel/Read/ReadVariableOp!dense_66/bias/Read/ReadVariableOp#dense_67/kernel/Read/ReadVariableOp!dense_67/bias/Read/ReadVariableOp#dense_68/kernel/Read/ReadVariableOp!dense_68/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_22/lstm_cell_22/kernel/Read/ReadVariableOp9lstm_22/lstm_cell_22/recurrent_kernel/Read/ReadVariableOp-lstm_22/lstm_cell_22/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_66/kernel/m/Read/ReadVariableOp(Adam/dense_66/bias/m/Read/ReadVariableOp*Adam/dense_67/kernel/m/Read/ReadVariableOp(Adam/dense_67/bias/m/Read/ReadVariableOp*Adam/dense_68/kernel/m/Read/ReadVariableOp(Adam/dense_68/bias/m/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_22/kernel/m/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_22/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_22/bias/m/Read/ReadVariableOp*Adam/dense_66/kernel/v/Read/ReadVariableOp(Adam/dense_66/bias/v/Read/ReadVariableOp*Adam/dense_67/kernel/v/Read/ReadVariableOp(Adam/dense_67/bias/v/Read/ReadVariableOp*Adam/dense_68/kernel/v/Read/ReadVariableOp(Adam/dense_68/bias/v/Read/ReadVariableOp6Adam/lstm_22/lstm_cell_22/kernel/v/Read/ReadVariableOp@Adam/lstm_22/lstm_cell_22/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_22/lstm_cell_22/bias/v/Read/ReadVariableOpConst*1
Tin*
(2&	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *(
f#R!
__inference__traced_save_444660
┌
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_66/kerneldense_66/biasdense_67/kerneldense_67/biasdense_68/kerneldense_68/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_22/lstm_cell_22/kernel%lstm_22/lstm_cell_22/recurrent_kernellstm_22/lstm_cell_22/biastotalcounttotal_1count_1Adam/dense_66/kernel/mAdam/dense_66/bias/mAdam/dense_67/kernel/mAdam/dense_67/bias/mAdam/dense_68/kernel/mAdam/dense_68/bias/m"Adam/lstm_22/lstm_cell_22/kernel/m,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m Adam/lstm_22/lstm_cell_22/bias/mAdam/dense_66/kernel/vAdam/dense_66/bias/vAdam/dense_67/kernel/vAdam/dense_67/bias/vAdam/dense_68/kernel/vAdam/dense_68/bias/v"Adam/lstm_22/lstm_cell_22/kernel/v,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v Adam/lstm_22/lstm_cell_22/bias/v*0
Tin)
'2%*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8ѓ *+
f&R$
"__inference__traced_restore_444778вЎ"
╝
Б
I__inference_sequential_21_layer_call_and_return_conditional_losses_442125
dense_66_input!
dense_66_442102:
dense_66_442104: 
lstm_22_442107:
lstm_22_442109: 
lstm_22_442111:!
dense_67_442114:
dense_67_442116:!
dense_68_442119:
dense_68_442121:
identityѕб dense_66/StatefulPartitionedCallб dense_67/StatefulPartitionedCallб dense_68/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallЧ
 dense_66/StatefulPartitionedCallStatefulPartitionedCalldense_66_inputdense_66_442102dense_66_442104*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_66_layer_call_and_return_conditional_losses_441271А
lstm_22/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0lstm_22_442107lstm_22_442109lstm_22_442111*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_441519њ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_67_442114dense_67_442116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_67_layer_call_and_return_conditional_losses_441537Њ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_442119dense_68_442121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_441553x
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Л
NoOpNoOp!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_namedense_66_input
╦
ч
D__inference_dense_66_layer_call_and_return_conditional_losses_441271

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ю
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_440850
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_440850___redundant_placeholder04
0while_while_cond_440850___redundant_placeholder14
0while_while_cond_440850___redundant_placeholder24
0while_while_cond_440850___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ќ

с
lstm_22_while_cond_442682,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1D
@lstm_22_while_lstm_22_while_cond_442682___redundant_placeholder0D
@lstm_22_while_lstm_22_while_cond_442682___redundant_placeholder1D
@lstm_22_while_lstm_22_while_cond_442682___redundant_placeholder2D
@lstm_22_while_lstm_22_while_cond_442682___redundant_placeholder3
lstm_22_while_identity
ѓ
lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: [
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ї
┤
(__inference_lstm_22_layer_call_fn_442968
inputs_0
unknown:
	unknown_0:
	unknown_1:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_440920o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
З
▓
(__inference_lstm_22_layer_call_fn_443001

inputs
unknown:
	unknown_0:
	unknown_1:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_441985o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
І8
ѓ
C__inference_lstm_22_layer_call_and_return_conditional_losses_440920

inputs%
lstm_cell_22_440838:!
lstm_cell_22_440840:%
lstm_cell_22_440842:
identityѕб$lstm_cell_22/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskш
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_440838lstm_cell_22_440840lstm_cell_22_440842*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_440837n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_440838lstm_cell_22_440840lstm_cell_22_440842*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_440851*
condR
while_cond_440850*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         u
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
х
├
while_cond_443109
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_443109___redundant_placeholder04
0while_while_cond_443109___redundant_placeholder14
0while_while_cond_443109___redundant_placeholder24
0while_while_cond_443109___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ёv
ъ	
while_body_443110
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_22_split_readvariableop_resource_0:B
4while_lstm_cell_22_split_1_readvariableop_resource_0:>
,while_lstm_cell_22_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_22_split_readvariableop_resource:@
2while_lstm_cell_22_split_1_readvariableop_resource:<
*while_lstm_cell_22_readvariableop_resource:ѕб!while/lstm_cell_22/ReadVariableOpб#while/lstm_cell_22/ReadVariableOp_1б#while/lstm_cell_22/ReadVariableOp_2б#while/lstm_cell_22/ReadVariableOp_3б'while/lstm_cell_22/split/ReadVariableOpб)while/lstm_cell_22/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ѓ
"while/lstm_cell_22/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?░
while/lstm_cell_22/ones_likeFill+while/lstm_cell_22/ones_like/Shape:output:0+while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         g
$while/lstm_cell_22/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
while/lstm_cell_22/ones_like_1Fill-while/lstm_cell_22/ones_like_1/Shape:output:0-while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         е
while/lstm_cell_22/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :џ
'while/lstm_cell_22/split/ReadVariableOpReadVariableOp2while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0О
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitћ
while/lstm_cell_22/MatMulMatMulwhile/lstm_cell_22/mul:z:0!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_1MatMulwhile/lstm_cell_22/mul_1:z:0!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_2MatMulwhile/lstm_cell_22/mul_2:z:0!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_3MatMulwhile/lstm_cell_22/mul_3:z:0!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         f
$while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : џ
)while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0═
while/lstm_cell_22/split_1Split-while/lstm_cell_22/split_1/split_dim:output:01while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitА
while/lstm_cell_22/BiasAddBiasAdd#while/lstm_cell_22/MatMul:product:0#while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_1BiasAdd%while/lstm_cell_22/MatMul_1:product:0#while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_2BiasAdd%while/lstm_cell_22/MatMul_2:product:0#while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_3BiasAdd%while/lstm_cell_22/MatMul_3:product:0#while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_4Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_5Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_6Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_7Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ј
!while/lstm_cell_22/ReadVariableOpReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_22/strided_sliceStridedSlice)while/lstm_cell_22/ReadVariableOp:value:0/while/lstm_cell_22/strided_slice/stack:output:01while/lstm_cell_22/strided_slice/stack_1:output:01while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskа
while/lstm_cell_22/MatMul_4MatMulwhile/lstm_cell_22/mul_4:z:0)while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/BiasAdd:output:0%while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         s
while/lstm_cell_22/SigmoidSigmoidwhile/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_1ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_1StridedSlice+while/lstm_cell_22/ReadVariableOp_1:value:01while/lstm_cell_22/strided_slice_1/stack:output:03while/lstm_cell_22/strided_slice_1/stack_1:output:03while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_5MatMulwhile/lstm_cell_22/mul_5:z:0+while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_1AddV2%while/lstm_cell_22/BiasAdd_1:output:0%while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_1Sigmoidwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ѕ
while/lstm_cell_22/mul_8Mul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_2ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_2StridedSlice+while/lstm_cell_22/ReadVariableOp_2:value:01while/lstm_cell_22/strided_slice_2/stack:output:03while/lstm_cell_22/strided_slice_2/stack_1:output:03while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_6MatMulwhile/lstm_cell_22/mul_6:z:0+while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_2AddV2%while/lstm_cell_22/BiasAdd_2:output:0%while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_2Sigmoidwhile/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell_22/mul_9Mulwhile/lstm_cell_22/Sigmoid:y:0 while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/add_3AddV2while/lstm_cell_22/mul_8:z:0while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_3ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_3StridedSlice+while/lstm_cell_22/ReadVariableOp_3:value:01while/lstm_cell_22/strided_slice_3/stack:output:03while/lstm_cell_22/strided_slice_3/stack_1:output:03while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_7MatMulwhile/lstm_cell_22/mul_7:z:0+while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_4AddV2%while/lstm_cell_22/BiasAdd_3:output:0%while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_3Sigmoidwhile/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_4Sigmoidwhile/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ќ
while/lstm_cell_22/mul_10Mul while/lstm_cell_22/Sigmoid_3:y:0 while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         к
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмz
while/Identity_4Identitywhile/lstm_cell_22/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         И

while/NoOpNoOp"^while/lstm_cell_22/ReadVariableOp$^while/lstm_cell_22/ReadVariableOp_1$^while/lstm_cell_22/ReadVariableOp_2$^while/lstm_cell_22/ReadVariableOp_3(^while/lstm_cell_22/split/ReadVariableOp*^while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_22_readvariableop_resource,while_lstm_cell_22_readvariableop_resource_0"j
2while_lstm_cell_22_split_1_readvariableop_resource4while_lstm_cell_22_split_1_readvariableop_resource_0"f
0while_lstm_cell_22_split_readvariableop_resource2while_lstm_cell_22_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2F
!while/lstm_cell_22/ReadVariableOp!while/lstm_cell_22/ReadVariableOp2J
#while/lstm_cell_22/ReadVariableOp_1#while/lstm_cell_22/ReadVariableOp_12J
#while/lstm_cell_22/ReadVariableOp_2#while/lstm_cell_22/ReadVariableOp_22J
#while/lstm_cell_22/ReadVariableOp_3#while/lstm_cell_22/ReadVariableOp_32R
'while/lstm_cell_22/split/ReadVariableOp'while/lstm_cell_22/split/ReadVariableOp2V
)while/lstm_cell_22/split_1/ReadVariableOp)while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ЮD
д
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_440837

inputs

states
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:         X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:         Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:         Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:         Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0ъ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         _
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         _
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         _
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         \
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:         \
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:         \
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:         \
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:         f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:         Z
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:         V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         Q
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:         ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         [

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
х
├
while_cond_441384
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_441384___redundant_placeholder04
0while_while_cond_441384___redundant_placeholder14
0while_while_cond_441384___redundant_placeholder24
0while_while_cond_441384___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ИЄ
Х
lstm_22_while_body_442338,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0L
:lstm_22_while_lstm_cell_22_split_readvariableop_resource_0:J
<lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0:F
4lstm_22_while_lstm_cell_22_readvariableop_resource_0:
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorJ
8lstm_22_while_lstm_cell_22_split_readvariableop_resource:H
:lstm_22_while_lstm_cell_22_split_1_readvariableop_resource:D
2lstm_22_while_lstm_cell_22_readvariableop_resource:ѕб)lstm_22/while/lstm_cell_22/ReadVariableOpб+lstm_22/while/lstm_cell_22/ReadVariableOp_1б+lstm_22/while/lstm_cell_22/ReadVariableOp_2б+lstm_22/while/lstm_cell_22/ReadVariableOp_3б/lstm_22/while/lstm_cell_22/split/ReadVariableOpб1lstm_22/while/lstm_cell_22/split_1/ReadVariableOpљ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
*lstm_22/while/lstm_cell_22/ones_like/ShapeShape8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:o
*lstm_22/while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?╚
$lstm_22/while/lstm_cell_22/ones_likeFill3lstm_22/while/lstm_cell_22/ones_like/Shape:output:03lstm_22/while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         w
,lstm_22/while/lstm_cell_22/ones_like_1/ShapeShapelstm_22_while_placeholder_2*
T0*
_output_shapes
:q
,lstm_22/while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?╬
&lstm_22/while/lstm_cell_22/ones_like_1Fill5lstm_22/while/lstm_cell_22/ones_like_1/Shape:output:05lstm_22/while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         └
lstm_22/while/lstm_cell_22/mulMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ┬
 lstm_22/while/lstm_cell_22/mul_1Mul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ┬
 lstm_22/while/lstm_cell_22/mul_2Mul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ┬
 lstm_22/while/lstm_cell_22/mul_3Mul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         l
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ф
/lstm_22/while/lstm_cell_22/split/ReadVariableOpReadVariableOp:lstm_22_while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0№
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:07lstm_22/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitг
!lstm_22/while/lstm_cell_22/MatMulMatMul"lstm_22/while/lstm_cell_22/mul:z:0)lstm_22/while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ░
#lstm_22/while/lstm_cell_22/MatMul_1MatMul$lstm_22/while/lstm_cell_22/mul_1:z:0)lstm_22/while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ░
#lstm_22/while/lstm_cell_22/MatMul_2MatMul$lstm_22/while/lstm_cell_22/mul_2:z:0)lstm_22/while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ░
#lstm_22/while/lstm_cell_22/MatMul_3MatMul$lstm_22/while/lstm_cell_22/mul_3:z:0)lstm_22/while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         n
,lstm_22/while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ф
1lstm_22/while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0т
"lstm_22/while/lstm_cell_22/split_1Split5lstm_22/while/lstm_cell_22/split_1/split_dim:output:09lstm_22/while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split╣
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd+lstm_22/while/lstm_cell_22/MatMul:product:0+lstm_22/while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         й
$lstm_22/while/lstm_cell_22/BiasAdd_1BiasAdd-lstm_22/while/lstm_cell_22/MatMul_1:product:0+lstm_22/while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         й
$lstm_22/while/lstm_cell_22/BiasAdd_2BiasAdd-lstm_22/while/lstm_cell_22/MatMul_2:product:0+lstm_22/while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         й
$lstm_22/while/lstm_cell_22/BiasAdd_3BiasAdd-lstm_22/while/lstm_cell_22/MatMul_3:product:0+lstm_22/while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         Д
 lstm_22/while/lstm_cell_22/mul_4Mullstm_22_while_placeholder_2/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Д
 lstm_22/while/lstm_cell_22/mul_5Mullstm_22_while_placeholder_2/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Д
 lstm_22/while/lstm_cell_22/mul_6Mullstm_22_while_placeholder_2/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Д
 lstm_22/while/lstm_cell_22/mul_7Mullstm_22_while_placeholder_2/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ъ
)lstm_22/while/lstm_cell_22/ReadVariableOpReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0
.lstm_22/while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Ђ
0lstm_22/while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ђ
0lstm_22/while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ы
(lstm_22/while/lstm_cell_22/strided_sliceStridedSlice1lstm_22/while/lstm_cell_22/ReadVariableOp:value:07lstm_22/while/lstm_cell_22/strided_slice/stack:output:09lstm_22/while/lstm_cell_22/strided_slice/stack_1:output:09lstm_22/while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskИ
#lstm_22/while/lstm_cell_22/MatMul_4MatMul$lstm_22/while/lstm_cell_22/mul_4:z:01lstm_22/while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         х
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/BiasAdd:output:0-lstm_22/while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         Ѓ
"lstm_22/while/lstm_cell_22/SigmoidSigmoid"lstm_22/while/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         а
+lstm_22/while/lstm_cell_22/ReadVariableOp_1ReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ђ
0lstm_22/while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ч
*lstm_22/while/lstm_cell_22/strided_slice_1StridedSlice3lstm_22/while/lstm_cell_22/ReadVariableOp_1:value:09lstm_22/while/lstm_cell_22/strided_slice_1/stack:output:0;lstm_22/while/lstm_cell_22/strided_slice_1/stack_1:output:0;lstm_22/while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask║
#lstm_22/while/lstm_cell_22/MatMul_5MatMul$lstm_22/while/lstm_cell_22/mul_5:z:03lstm_22/while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         ╣
 lstm_22/while/lstm_cell_22/add_1AddV2-lstm_22/while/lstm_cell_22/BiasAdd_1:output:0-lstm_22/while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         а
 lstm_22/while/lstm_cell_22/mul_8Mul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*'
_output_shapes
:         а
+lstm_22/while/lstm_cell_22/ReadVariableOp_2ReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ђ
0lstm_22/while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ч
*lstm_22/while/lstm_cell_22/strided_slice_2StridedSlice3lstm_22/while/lstm_cell_22/ReadVariableOp_2:value:09lstm_22/while/lstm_cell_22/strided_slice_2/stack:output:0;lstm_22/while/lstm_cell_22/strided_slice_2/stack_1:output:0;lstm_22/while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask║
#lstm_22/while/lstm_cell_22/MatMul_6MatMul$lstm_22/while/lstm_cell_22/mul_6:z:03lstm_22/while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         ╣
 lstm_22/while/lstm_cell_22/add_2AddV2-lstm_22/while/lstm_cell_22/BiasAdd_2:output:0-lstm_22/while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid$lstm_22/while/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ф
 lstm_22/while/lstm_cell_22/mul_9Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Д
 lstm_22/while/lstm_cell_22/add_3AddV2$lstm_22/while/lstm_cell_22/mul_8:z:0$lstm_22/while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         а
+lstm_22/while/lstm_cell_22/ReadVariableOp_3ReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ђ
0lstm_22/while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ч
*lstm_22/while/lstm_cell_22/strided_slice_3StridedSlice3lstm_22/while/lstm_cell_22/ReadVariableOp_3:value:09lstm_22/while/lstm_cell_22/strided_slice_3/stack:output:0;lstm_22/while/lstm_cell_22/strided_slice_3/stack_1:output:0;lstm_22/while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask║
#lstm_22/while/lstm_cell_22/MatMul_7MatMul$lstm_22/while/lstm_cell_22/mul_7:z:03lstm_22/while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         ╣
 lstm_22/while/lstm_cell_22/add_4AddV2-lstm_22/while/lstm_cell_22/BiasAdd_3:output:0-lstm_22/while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_3Sigmoid$lstm_22/while/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_4Sigmoid$lstm_22/while/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         «
!lstm_22/while/lstm_cell_22/mul_10Mul(lstm_22/while/lstm_cell_22/Sigmoid_3:y:0(lstm_22/while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         Т
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder%lstm_22/while/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: і
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: q
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: ▒
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: :жУмњ
lstm_22/while/Identity_4Identity%lstm_22/while/lstm_cell_22/mul_10:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_3:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_22/while/NoOpNoOp*^lstm_22/while/lstm_cell_22/ReadVariableOp,^lstm_22/while/lstm_cell_22/ReadVariableOp_1,^lstm_22/while/lstm_cell_22/ReadVariableOp_2,^lstm_22/while/lstm_cell_22/ReadVariableOp_30^lstm_22/while/lstm_cell_22/split/ReadVariableOp2^lstm_22/while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"j
2lstm_22_while_lstm_cell_22_readvariableop_resource4lstm_22_while_lstm_cell_22_readvariableop_resource_0"z
:lstm_22_while_lstm_cell_22_split_1_readvariableop_resource<lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0"v
8lstm_22_while_lstm_cell_22_split_readvariableop_resource:lstm_22_while_lstm_cell_22_split_readvariableop_resource_0"╚
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)lstm_22/while/lstm_cell_22/ReadVariableOp)lstm_22/while/lstm_cell_22/ReadVariableOp2Z
+lstm_22/while/lstm_cell_22/ReadVariableOp_1+lstm_22/while/lstm_cell_22/ReadVariableOp_12Z
+lstm_22/while/lstm_cell_22/ReadVariableOp_2+lstm_22/while/lstm_cell_22/ReadVariableOp_22Z
+lstm_22/while/lstm_cell_22/ReadVariableOp_3+lstm_22/while/lstm_cell_22/ReadVariableOp_32b
/lstm_22/while/lstm_cell_22/split/ReadVariableOp/lstm_22/while/lstm_cell_22/split/ReadVariableOp2f
1lstm_22/while/lstm_cell_22/split_1/ReadVariableOp1lstm_22/while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
К	
ш
D__inference_dense_68_layer_call_and_return_conditional_losses_444267

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ќ

с
lstm_22_while_cond_442337,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3.
*lstm_22_while_less_lstm_22_strided_slice_1D
@lstm_22_while_lstm_22_while_cond_442337___redundant_placeholder0D
@lstm_22_while_lstm_22_while_cond_442337___redundant_placeholder1D
@lstm_22_while_lstm_22_while_cond_442337___redundant_placeholder2D
@lstm_22_while_lstm_22_while_cond_442337___redundant_placeholder3
lstm_22_while_identity
ѓ
lstm_22/while/LessLesslstm_22_while_placeholder*lstm_22_while_less_lstm_22_strided_slice_1*
T0*
_output_shapes
: [
lstm_22/while/IdentityIdentitylstm_22/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_22_while_identitylstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
з"
П
while_body_441156
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_22_441180_0:)
while_lstm_cell_22_441182_0:-
while_lstm_cell_22_441184_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_22_441180:'
while_lstm_cell_22_441182:+
while_lstm_cell_22_441184:ѕб*while/lstm_cell_22/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0│
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_441180_0while_lstm_cell_22_441182_0while_lstm_cell_22_441184_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_441097▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмљ
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_22_441180while_lstm_cell_22_441180_0"8
while_lstm_cell_22_441182while_lstm_cell_22_441182_0"8
while_lstm_cell_22_441184while_lstm_cell_22_441184_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ї
┤
(__inference_lstm_22_layer_call_fn_442979
inputs_0
unknown:
	unknown_0:
	unknown_1:
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_441225o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
К	
ш
D__inference_dense_67_layer_call_and_return_conditional_losses_444248

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▀Ђ
С
C__inference_lstm_22_layer_call_and_return_conditional_losses_441519

inputs<
*lstm_cell_22_split_readvariableop_resource::
,lstm_cell_22_split_1_readvariableop_resource:6
$lstm_cell_22_readvariableop_resource:
identityѕбlstm_cell_22/ReadVariableOpбlstm_cell_22/ReadVariableOp_1бlstm_cell_22/ReadVariableOp_2бlstm_cell_22/ReadVariableOp_3б!lstm_cell_22/split/ReadVariableOpб#lstm_cell_22/split_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskd
lstm_cell_22/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ъ
lstm_cell_22/ones_likeFill%lstm_cell_22/ones_like/Shape:output:0%lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         \
lstm_cell_22/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ц
lstm_cell_22/ones_like_1Fill'lstm_cell_22/ones_like_1/Shape:output:0'lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mulMulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_2Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_3Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
!lstm_cell_22/split/ReadVariableOpReadVariableOp*lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0┼
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0)lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitѓ
lstm_cell_22/MatMulMatMullstm_cell_22/mul:z:0lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_1MatMullstm_cell_22/mul_1:z:0lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_2MatMullstm_cell_22/mul_2:z:0lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_3MatMullstm_cell_22/mul_3:z:0lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         `
lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ї
#lstm_cell_22/split_1/ReadVariableOpReadVariableOp,lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0╗
lstm_cell_22/split_1Split'lstm_cell_22/split_1/split_dim:output:0+lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЈ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/MatMul:product:0lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_1BiasAddlstm_cell_22/MatMul_1:product:0lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_2BiasAddlstm_cell_22/MatMul_2:product:0lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_3BiasAddlstm_cell_22/MatMul_3:product:0lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_4Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_5Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_6Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_7Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ђ
lstm_cell_22/ReadVariableOpReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      г
lstm_cell_22/strided_sliceStridedSlice#lstm_cell_22/ReadVariableOp:value:0)lstm_cell_22/strided_slice/stack:output:0+lstm_cell_22/strided_slice/stack_1:output:0+lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskј
lstm_cell_22/MatMul_4MatMullstm_cell_22/mul_4:z:0#lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         І
lstm_cell_22/addAddV2lstm_cell_22/BiasAdd:output:0lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         g
lstm_cell_22/SigmoidSigmoidlstm_cell_22/add:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_1ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_1StridedSlice%lstm_cell_22/ReadVariableOp_1:value:0+lstm_cell_22/strided_slice_1/stack:output:0-lstm_cell_22/strided_slice_1/stack_1:output:0-lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_5MatMullstm_cell_22/mul_5:z:0%lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_1AddV2lstm_cell_22/BiasAdd_1:output:0lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         y
lstm_cell_22/mul_8Mullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_2ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_2StridedSlice%lstm_cell_22/ReadVariableOp_2:value:0+lstm_cell_22/strided_slice_2/stack:output:0-lstm_cell_22/strided_slice_2/stack_1:output:0-lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_6MatMullstm_cell_22/mul_6:z:0%lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_2AddV2lstm_cell_22/BiasAdd_2:output:0lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell_22/mul_9Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         }
lstm_cell_22/add_3AddV2lstm_cell_22/mul_8:z:0lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_3ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_3StridedSlice%lstm_cell_22/ReadVariableOp_3:value:0+lstm_cell_22/strided_slice_3/stack:output:0-lstm_cell_22/strided_slice_3/stack_1:output:0-lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_7MatMullstm_cell_22/mul_7:z:0%lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_4AddV2lstm_cell_22/BiasAdd_3:output:0lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_3Sigmoidlstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_4Sigmoidlstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mul_10Mullstm_cell_22/Sigmoid_3:y:0lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_22_split_readvariableop_resource,lstm_cell_22_split_1_readvariableop_resource$lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_441385*
condR
while_cond_441384*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp^lstm_cell_22/ReadVariableOp^lstm_cell_22/ReadVariableOp_1^lstm_cell_22/ReadVariableOp_2^lstm_cell_22/ReadVariableOp_3"^lstm_cell_22/split/ReadVariableOp$^lstm_cell_22/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2:
lstm_cell_22/ReadVariableOplstm_cell_22/ReadVariableOp2>
lstm_cell_22/ReadVariableOp_1lstm_cell_22/ReadVariableOp_12>
lstm_cell_22/ReadVariableOp_2lstm_cell_22/ReadVariableOp_22>
lstm_cell_22/ReadVariableOp_3lstm_cell_22/ReadVariableOp_32F
!lstm_cell_22/split/ReadVariableOp!lstm_cell_22/split/ReadVariableOp2J
#lstm_cell_22/split_1/ReadVariableOp#lstm_cell_22/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
У
з
-__inference_lstm_cell_22_layer_call_fn_444301

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2ѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_441097o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
ћ

Р
.__inference_sequential_21_layer_call_fn_441581
dense_66_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCalldense_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_441560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_namedense_66_input
╦
ч
D__inference_dense_66_layer_call_and_return_conditional_losses_442957

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       E
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ┐
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ю
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:         і
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  і
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         [
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Д
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         z
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
К	
ш
D__inference_dense_68_layer_call_and_return_conditional_losses_441553

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ц
Џ
I__inference_sequential_21_layer_call_and_return_conditional_losses_442055

inputs!
dense_66_442032:
dense_66_442034: 
lstm_22_442037:
lstm_22_442039: 
lstm_22_442041:!
dense_67_442044:
dense_67_442046:!
dense_68_442049:
dense_68_442051:
identityѕб dense_66/StatefulPartitionedCallб dense_67/StatefulPartitionedCallб dense_68/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallЗ
 dense_66/StatefulPartitionedCallStatefulPartitionedCallinputsdense_66_442032dense_66_442034*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_66_layer_call_and_return_conditional_losses_441271А
lstm_22/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0lstm_22_442037lstm_22_442039lstm_22_442041*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_441985њ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_67_442044dense_67_442046*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_67_layer_call_and_return_conditional_losses_441537Њ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_442049dense_68_442051*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_441553x
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Л
NoOpNoOp!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
й~
д
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_441097

inputs

states
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:         O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:љ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:љ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:љ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:         G
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:         T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:         S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:         S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:         S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:         S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:         W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0ъ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         _
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         _
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         _
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         [
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:         Z
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:         V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         Q
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:         ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         [

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_namestates:OK
'
_output_shapes
:         
 
_user_specified_namestates
┬
ќ
)__inference_dense_68_layer_call_fn_444257

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_441553o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
з"
П
while_body_440851
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_22_440875_0:)
while_lstm_cell_22_440877_0:-
while_lstm_cell_22_440879_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_22_440875:'
while_lstm_cell_22_440877:+
while_lstm_cell_22_440879:ѕб*while/lstm_cell_22/StatefulPartitionedCallѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0│
*while/lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_22_440875_0while_lstm_cell_22_440877_0while_lstm_cell_22_440879_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_440837▄
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_22/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмљ
while/Identity_4Identity3while/lstm_cell_22/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:         љ
while/Identity_5Identity3while/lstm_cell_22/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:         y

while/NoOpNoOp+^while/lstm_cell_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_22_440875while_lstm_cell_22_440875_0"8
while_lstm_cell_22_440877while_lstm_cell_22_440877_0"8
while_lstm_cell_22_440879while_lstm_cell_22_440879_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2X
*while/lstm_cell_22/StatefulPartitionedCall*while/lstm_cell_22/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ля
Х
lstm_22_while_body_442683,
(lstm_22_while_lstm_22_while_loop_counter2
.lstm_22_while_lstm_22_while_maximum_iterations
lstm_22_while_placeholder
lstm_22_while_placeholder_1
lstm_22_while_placeholder_2
lstm_22_while_placeholder_3+
'lstm_22_while_lstm_22_strided_slice_1_0g
clstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0L
:lstm_22_while_lstm_cell_22_split_readvariableop_resource_0:J
<lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0:F
4lstm_22_while_lstm_cell_22_readvariableop_resource_0:
lstm_22_while_identity
lstm_22_while_identity_1
lstm_22_while_identity_2
lstm_22_while_identity_3
lstm_22_while_identity_4
lstm_22_while_identity_5)
%lstm_22_while_lstm_22_strided_slice_1e
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorJ
8lstm_22_while_lstm_cell_22_split_readvariableop_resource:H
:lstm_22_while_lstm_cell_22_split_1_readvariableop_resource:D
2lstm_22_while_lstm_cell_22_readvariableop_resource:ѕб)lstm_22/while/lstm_cell_22/ReadVariableOpб+lstm_22/while/lstm_cell_22/ReadVariableOp_1б+lstm_22/while/lstm_cell_22/ReadVariableOp_2б+lstm_22/while/lstm_cell_22/ReadVariableOp_3б/lstm_22/while/lstm_cell_22/split/ReadVariableOpб1lstm_22/while/lstm_cell_22/split_1/ReadVariableOpљ
?lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╬
1lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0lstm_22_while_placeholderHlstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0њ
*lstm_22/while/lstm_cell_22/ones_like/ShapeShape8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:o
*lstm_22/while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?╚
$lstm_22/while/lstm_cell_22/ones_likeFill3lstm_22/while/lstm_cell_22/ones_like/Shape:output:03lstm_22/while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         m
(lstm_22/while/lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?┴
&lstm_22/while/lstm_cell_22/dropout/MulMul-lstm_22/while/lstm_cell_22/ones_like:output:01lstm_22/while/lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         Ё
(lstm_22/while/lstm_cell_22/dropout/ShapeShape-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:┬
?lstm_22/while/lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform1lstm_22/while/lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0v
1lstm_22/while/lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=э
/lstm_22/while/lstm_cell_22/dropout/GreaterEqualGreaterEqualHlstm_22/while/lstm_cell_22/dropout/random_uniform/RandomUniform:output:0:lstm_22/while/lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ц
'lstm_22/while/lstm_cell_22/dropout/CastCast3lstm_22/while/lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ║
(lstm_22/while/lstm_cell_22/dropout/Mul_1Mul*lstm_22/while/lstm_cell_22/dropout/Mul:z:0+lstm_22/while/lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         o
*lstm_22/while/lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?┼
(lstm_22/while/lstm_cell_22/dropout_1/MulMul-lstm_22/while/lstm_cell_22/ones_like:output:03lstm_22/while/lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         Є
*lstm_22/while/lstm_cell_22/dropout_1/ShapeShape-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:к
Alstm_22/while/lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform3lstm_22/while/lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0x
3lstm_22/while/lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=§
1lstm_22/while/lstm_cell_22/dropout_1/GreaterEqualGreaterEqualJlstm_22/while/lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:0<lstm_22/while/lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
)lstm_22/while/lstm_cell_22/dropout_1/CastCast5lstm_22/while/lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         └
*lstm_22/while/lstm_cell_22/dropout_1/Mul_1Mul,lstm_22/while/lstm_cell_22/dropout_1/Mul:z:0-lstm_22/while/lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         o
*lstm_22/while/lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?┼
(lstm_22/while/lstm_cell_22/dropout_2/MulMul-lstm_22/while/lstm_cell_22/ones_like:output:03lstm_22/while/lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         Є
*lstm_22/while/lstm_cell_22/dropout_2/ShapeShape-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:к
Alstm_22/while/lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform3lstm_22/while/lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0x
3lstm_22/while/lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=§
1lstm_22/while/lstm_cell_22/dropout_2/GreaterEqualGreaterEqualJlstm_22/while/lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:0<lstm_22/while/lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
)lstm_22/while/lstm_cell_22/dropout_2/CastCast5lstm_22/while/lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         └
*lstm_22/while/lstm_cell_22/dropout_2/Mul_1Mul,lstm_22/while/lstm_cell_22/dropout_2/Mul:z:0-lstm_22/while/lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         o
*lstm_22/while/lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?┼
(lstm_22/while/lstm_cell_22/dropout_3/MulMul-lstm_22/while/lstm_cell_22/ones_like:output:03lstm_22/while/lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         Є
*lstm_22/while/lstm_cell_22/dropout_3/ShapeShape-lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:к
Alstm_22/while/lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform3lstm_22/while/lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0x
3lstm_22/while/lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=§
1lstm_22/while/lstm_cell_22/dropout_3/GreaterEqualGreaterEqualJlstm_22/while/lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:0<lstm_22/while/lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
)lstm_22/while/lstm_cell_22/dropout_3/CastCast5lstm_22/while/lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         └
*lstm_22/while/lstm_cell_22/dropout_3/Mul_1Mul,lstm_22/while/lstm_cell_22/dropout_3/Mul:z:0-lstm_22/while/lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         w
,lstm_22/while/lstm_cell_22/ones_like_1/ShapeShapelstm_22_while_placeholder_2*
T0*
_output_shapes
:q
,lstm_22/while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?╬
&lstm_22/while/lstm_cell_22/ones_like_1Fill5lstm_22/while/lstm_cell_22/ones_like_1/Shape:output:05lstm_22/while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         o
*lstm_22/while/lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?К
(lstm_22/while/lstm_cell_22/dropout_4/MulMul/lstm_22/while/lstm_cell_22/ones_like_1:output:03lstm_22/while/lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         Ѕ
*lstm_22/while/lstm_cell_22/dropout_4/ShapeShape/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:к
Alstm_22/while/lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform3lstm_22/while/lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0x
3lstm_22/while/lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=§
1lstm_22/while/lstm_cell_22/dropout_4/GreaterEqualGreaterEqualJlstm_22/while/lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:0<lstm_22/while/lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
)lstm_22/while/lstm_cell_22/dropout_4/CastCast5lstm_22/while/lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         └
*lstm_22/while/lstm_cell_22/dropout_4/Mul_1Mul,lstm_22/while/lstm_cell_22/dropout_4/Mul:z:0-lstm_22/while/lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         o
*lstm_22/while/lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?К
(lstm_22/while/lstm_cell_22/dropout_5/MulMul/lstm_22/while/lstm_cell_22/ones_like_1:output:03lstm_22/while/lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         Ѕ
*lstm_22/while/lstm_cell_22/dropout_5/ShapeShape/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:к
Alstm_22/while/lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform3lstm_22/while/lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0x
3lstm_22/while/lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=§
1lstm_22/while/lstm_cell_22/dropout_5/GreaterEqualGreaterEqualJlstm_22/while/lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:0<lstm_22/while/lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
)lstm_22/while/lstm_cell_22/dropout_5/CastCast5lstm_22/while/lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         └
*lstm_22/while/lstm_cell_22/dropout_5/Mul_1Mul,lstm_22/while/lstm_cell_22/dropout_5/Mul:z:0-lstm_22/while/lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         o
*lstm_22/while/lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?К
(lstm_22/while/lstm_cell_22/dropout_6/MulMul/lstm_22/while/lstm_cell_22/ones_like_1:output:03lstm_22/while/lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         Ѕ
*lstm_22/while/lstm_cell_22/dropout_6/ShapeShape/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:к
Alstm_22/while/lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform3lstm_22/while/lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0x
3lstm_22/while/lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=§
1lstm_22/while/lstm_cell_22/dropout_6/GreaterEqualGreaterEqualJlstm_22/while/lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:0<lstm_22/while/lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
)lstm_22/while/lstm_cell_22/dropout_6/CastCast5lstm_22/while/lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         └
*lstm_22/while/lstm_cell_22/dropout_6/Mul_1Mul,lstm_22/while/lstm_cell_22/dropout_6/Mul:z:0-lstm_22/while/lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         o
*lstm_22/while/lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?К
(lstm_22/while/lstm_cell_22/dropout_7/MulMul/lstm_22/while/lstm_cell_22/ones_like_1:output:03lstm_22/while/lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         Ѕ
*lstm_22/while/lstm_cell_22/dropout_7/ShapeShape/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:к
Alstm_22/while/lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform3lstm_22/while/lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0x
3lstm_22/while/lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=§
1lstm_22/while/lstm_cell_22/dropout_7/GreaterEqualGreaterEqualJlstm_22/while/lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:0<lstm_22/while/lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Е
)lstm_22/while/lstm_cell_22/dropout_7/CastCast5lstm_22/while/lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         └
*lstm_22/while/lstm_cell_22/dropout_7/Mul_1Mul,lstm_22/while/lstm_cell_22/dropout_7/Mul:z:0-lstm_22/while/lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         ┐
lstm_22/while/lstm_cell_22/mulMul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_22/while/lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         ├
 lstm_22/while/lstm_cell_22/mul_1Mul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm_22/while/lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         ├
 lstm_22/while/lstm_cell_22/mul_2Mul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm_22/while/lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         ├
 lstm_22/while/lstm_cell_22/mul_3Mul8lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm_22/while/lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         l
*lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ф
/lstm_22/while/lstm_cell_22/split/ReadVariableOpReadVariableOp:lstm_22_while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0№
 lstm_22/while/lstm_cell_22/splitSplit3lstm_22/while/lstm_cell_22/split/split_dim:output:07lstm_22/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitг
!lstm_22/while/lstm_cell_22/MatMulMatMul"lstm_22/while/lstm_cell_22/mul:z:0)lstm_22/while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ░
#lstm_22/while/lstm_cell_22/MatMul_1MatMul$lstm_22/while/lstm_cell_22/mul_1:z:0)lstm_22/while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ░
#lstm_22/while/lstm_cell_22/MatMul_2MatMul$lstm_22/while/lstm_cell_22/mul_2:z:0)lstm_22/while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ░
#lstm_22/while/lstm_cell_22/MatMul_3MatMul$lstm_22/while/lstm_cell_22/mul_3:z:0)lstm_22/while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         n
,lstm_22/while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ф
1lstm_22/while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp<lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0т
"lstm_22/while/lstm_cell_22/split_1Split5lstm_22/while/lstm_cell_22/split_1/split_dim:output:09lstm_22/while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split╣
"lstm_22/while/lstm_cell_22/BiasAddBiasAdd+lstm_22/while/lstm_cell_22/MatMul:product:0+lstm_22/while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         й
$lstm_22/while/lstm_cell_22/BiasAdd_1BiasAdd-lstm_22/while/lstm_cell_22/MatMul_1:product:0+lstm_22/while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         й
$lstm_22/while/lstm_cell_22/BiasAdd_2BiasAdd-lstm_22/while/lstm_cell_22/MatMul_2:product:0+lstm_22/while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         й
$lstm_22/while/lstm_cell_22/BiasAdd_3BiasAdd-lstm_22/while/lstm_cell_22/MatMul_3:product:0+lstm_22/while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         д
 lstm_22/while/lstm_cell_22/mul_4Mullstm_22_while_placeholder_2.lstm_22/while/lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         д
 lstm_22/while/lstm_cell_22/mul_5Mullstm_22_while_placeholder_2.lstm_22/while/lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         д
 lstm_22/while/lstm_cell_22/mul_6Mullstm_22_while_placeholder_2.lstm_22/while/lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         д
 lstm_22/while/lstm_cell_22/mul_7Mullstm_22_while_placeholder_2.lstm_22/while/lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         ъ
)lstm_22/while/lstm_cell_22/ReadVariableOpReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0
.lstm_22/while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Ђ
0lstm_22/while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ђ
0lstm_22/while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ы
(lstm_22/while/lstm_cell_22/strided_sliceStridedSlice1lstm_22/while/lstm_cell_22/ReadVariableOp:value:07lstm_22/while/lstm_cell_22/strided_slice/stack:output:09lstm_22/while/lstm_cell_22/strided_slice/stack_1:output:09lstm_22/while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskИ
#lstm_22/while/lstm_cell_22/MatMul_4MatMul$lstm_22/while/lstm_cell_22/mul_4:z:01lstm_22/while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         х
lstm_22/while/lstm_cell_22/addAddV2+lstm_22/while/lstm_cell_22/BiasAdd:output:0-lstm_22/while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         Ѓ
"lstm_22/while/lstm_cell_22/SigmoidSigmoid"lstm_22/while/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         а
+lstm_22/while/lstm_cell_22/ReadVariableOp_1ReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ђ
0lstm_22/while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ч
*lstm_22/while/lstm_cell_22/strided_slice_1StridedSlice3lstm_22/while/lstm_cell_22/ReadVariableOp_1:value:09lstm_22/while/lstm_cell_22/strided_slice_1/stack:output:0;lstm_22/while/lstm_cell_22/strided_slice_1/stack_1:output:0;lstm_22/while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask║
#lstm_22/while/lstm_cell_22/MatMul_5MatMul$lstm_22/while/lstm_cell_22/mul_5:z:03lstm_22/while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         ╣
 lstm_22/while/lstm_cell_22/add_1AddV2-lstm_22/while/lstm_cell_22/BiasAdd_1:output:0-lstm_22/while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid$lstm_22/while/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         а
 lstm_22/while/lstm_cell_22/mul_8Mul(lstm_22/while/lstm_cell_22/Sigmoid_1:y:0lstm_22_while_placeholder_3*
T0*'
_output_shapes
:         а
+lstm_22/while/lstm_cell_22/ReadVariableOp_2ReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ђ
0lstm_22/while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ч
*lstm_22/while/lstm_cell_22/strided_slice_2StridedSlice3lstm_22/while/lstm_cell_22/ReadVariableOp_2:value:09lstm_22/while/lstm_cell_22/strided_slice_2/stack:output:0;lstm_22/while/lstm_cell_22/strided_slice_2/stack_1:output:0;lstm_22/while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask║
#lstm_22/while/lstm_cell_22/MatMul_6MatMul$lstm_22/while/lstm_cell_22/mul_6:z:03lstm_22/while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         ╣
 lstm_22/while/lstm_cell_22/add_2AddV2-lstm_22/while/lstm_cell_22/BiasAdd_2:output:0-lstm_22/while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid$lstm_22/while/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ф
 lstm_22/while/lstm_cell_22/mul_9Mul&lstm_22/while/lstm_cell_22/Sigmoid:y:0(lstm_22/while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Д
 lstm_22/while/lstm_cell_22/add_3AddV2$lstm_22/while/lstm_cell_22/mul_8:z:0$lstm_22/while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         а
+lstm_22/while/lstm_cell_22/ReadVariableOp_3ReadVariableOp4lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ђ
0lstm_22/while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Ѓ
2lstm_22/while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ч
*lstm_22/while/lstm_cell_22/strided_slice_3StridedSlice3lstm_22/while/lstm_cell_22/ReadVariableOp_3:value:09lstm_22/while/lstm_cell_22/strided_slice_3/stack:output:0;lstm_22/while/lstm_cell_22/strided_slice_3/stack_1:output:0;lstm_22/while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask║
#lstm_22/while/lstm_cell_22/MatMul_7MatMul$lstm_22/while/lstm_cell_22/mul_7:z:03lstm_22/while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         ╣
 lstm_22/while/lstm_cell_22/add_4AddV2-lstm_22/while/lstm_cell_22/BiasAdd_3:output:0-lstm_22/while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_3Sigmoid$lstm_22/while/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         Є
$lstm_22/while/lstm_cell_22/Sigmoid_4Sigmoid$lstm_22/while/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         «
!lstm_22/while/lstm_cell_22/mul_10Mul(lstm_22/while/lstm_cell_22/Sigmoid_3:y:0(lstm_22/while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         Т
2lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_22_while_placeholder_1lstm_22_while_placeholder%lstm_22/while/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмU
lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_22/while/addAddV2lstm_22_while_placeholderlstm_22/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :Є
lstm_22/while/add_1AddV2(lstm_22_while_lstm_22_while_loop_counterlstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_22/while/IdentityIdentitylstm_22/while/add_1:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: і
lstm_22/while/Identity_1Identity.lstm_22_while_lstm_22_while_maximum_iterations^lstm_22/while/NoOp*
T0*
_output_shapes
: q
lstm_22/while/Identity_2Identitylstm_22/while/add:z:0^lstm_22/while/NoOp*
T0*
_output_shapes
: ▒
lstm_22/while/Identity_3IdentityBlstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_22/while/NoOp*
T0*
_output_shapes
: :жУмњ
lstm_22/while/Identity_4Identity%lstm_22/while/lstm_cell_22/mul_10:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         Љ
lstm_22/while/Identity_5Identity$lstm_22/while/lstm_cell_22/add_3:z:0^lstm_22/while/NoOp*
T0*'
_output_shapes
:         ­
lstm_22/while/NoOpNoOp*^lstm_22/while/lstm_cell_22/ReadVariableOp,^lstm_22/while/lstm_cell_22/ReadVariableOp_1,^lstm_22/while/lstm_cell_22/ReadVariableOp_2,^lstm_22/while/lstm_cell_22/ReadVariableOp_30^lstm_22/while/lstm_cell_22/split/ReadVariableOp2^lstm_22/while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_22_while_identitylstm_22/while/Identity:output:0"=
lstm_22_while_identity_1!lstm_22/while/Identity_1:output:0"=
lstm_22_while_identity_2!lstm_22/while/Identity_2:output:0"=
lstm_22_while_identity_3!lstm_22/while/Identity_3:output:0"=
lstm_22_while_identity_4!lstm_22/while/Identity_4:output:0"=
lstm_22_while_identity_5!lstm_22/while/Identity_5:output:0"P
%lstm_22_while_lstm_22_strided_slice_1'lstm_22_while_lstm_22_strided_slice_1_0"j
2lstm_22_while_lstm_cell_22_readvariableop_resource4lstm_22_while_lstm_cell_22_readvariableop_resource_0"z
:lstm_22_while_lstm_cell_22_split_1_readvariableop_resource<lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0"v
8lstm_22_while_lstm_cell_22_split_readvariableop_resource:lstm_22_while_lstm_cell_22_split_readvariableop_resource_0"╚
alstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensorclstm_22_while_tensorarrayv2read_tensorlistgetitem_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2V
)lstm_22/while/lstm_cell_22/ReadVariableOp)lstm_22/while/lstm_cell_22/ReadVariableOp2Z
+lstm_22/while/lstm_cell_22/ReadVariableOp_1+lstm_22/while/lstm_cell_22/ReadVariableOp_12Z
+lstm_22/while/lstm_cell_22/ReadVariableOp_2+lstm_22/while/lstm_cell_22/ReadVariableOp_22Z
+lstm_22/while/lstm_cell_22/ReadVariableOp_3+lstm_22/while/lstm_cell_22/ReadVariableOp_32b
/lstm_22/while/lstm_cell_22/split/ReadVariableOp/lstm_22/while/lstm_cell_22/split/ReadVariableOp2f
1lstm_22/while/lstm_cell_22/split_1/ReadVariableOp1lstm_22/while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
 ╔
С
C__inference_lstm_22_layer_call_and_return_conditional_losses_444229

inputs<
*lstm_cell_22_split_readvariableop_resource::
,lstm_cell_22_split_1_readvariableop_resource:6
$lstm_cell_22_readvariableop_resource:
identityѕбlstm_cell_22/ReadVariableOpбlstm_cell_22/ReadVariableOp_1бlstm_cell_22/ReadVariableOp_2бlstm_cell_22/ReadVariableOp_3б!lstm_cell_22/split/ReadVariableOpб#lstm_cell_22/split_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskd
lstm_cell_22/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ъ
lstm_cell_22/ones_likeFill%lstm_cell_22/ones_like/Shape:output:0%lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         _
lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ќ
lstm_cell_22/dropout/MulMullstm_cell_22/ones_like:output:0#lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         i
lstm_cell_22/dropout/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:д
1lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0h
#lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=═
!lstm_cell_22/dropout/GreaterEqualGreaterEqual:lstm_cell_22/dropout/random_uniform/RandomUniform:output:0,lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ѕ
lstm_cell_22/dropout/CastCast%lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         љ
lstm_cell_22/dropout/Mul_1Mullstm_cell_22/dropout/Mul:z:0lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_1/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_1/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_1/GreaterEqualGreaterEqual<lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_1/CastCast'lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_1/Mul_1Mullstm_cell_22/dropout_1/Mul:z:0lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_2/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_2/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_2/GreaterEqualGreaterEqual<lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_2/CastCast'lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_2/Mul_1Mullstm_cell_22/dropout_2/Mul:z:0lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_3/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_3/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_3/GreaterEqualGreaterEqual<lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_3/CastCast'lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_3/Mul_1Mullstm_cell_22/dropout_3/Mul:z:0lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         \
lstm_cell_22/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ц
lstm_cell_22/ones_like_1Fill'lstm_cell_22/ones_like_1/Shape:output:0'lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_4/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_4/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_4/GreaterEqualGreaterEqual<lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_4/CastCast'lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_4/Mul_1Mullstm_cell_22/dropout_4/Mul:z:0lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_5/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_5/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_5/GreaterEqualGreaterEqual<lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_5/CastCast'lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_5/Mul_1Mullstm_cell_22/dropout_5/Mul:z:0lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_6/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_6/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_6/GreaterEqualGreaterEqual<lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_6/CastCast'lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_6/Mul_1Mullstm_cell_22/dropout_6/Mul:z:0lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_7/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_7/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_7/GreaterEqualGreaterEqual<lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_7/CastCast'lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_7/Mul_1Mullstm_cell_22/dropout_7/Mul:z:0lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         Ѓ
lstm_cell_22/mulMulstrided_slice_2:output:0lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_1Mulstrided_slice_2:output:0 lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_2Mulstrided_slice_2:output:0 lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_3Mulstrided_slice_2:output:0 lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         ^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
!lstm_cell_22/split/ReadVariableOpReadVariableOp*lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0┼
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0)lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitѓ
lstm_cell_22/MatMulMatMullstm_cell_22/mul:z:0lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_1MatMullstm_cell_22/mul_1:z:0lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_2MatMullstm_cell_22/mul_2:z:0lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_3MatMullstm_cell_22/mul_3:z:0lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         `
lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ї
#lstm_cell_22/split_1/ReadVariableOpReadVariableOp,lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0╗
lstm_cell_22/split_1Split'lstm_cell_22/split_1/split_dim:output:0+lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЈ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/MatMul:product:0lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_1BiasAddlstm_cell_22/MatMul_1:product:0lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_2BiasAddlstm_cell_22/MatMul_2:product:0lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_3BiasAddlstm_cell_22/MatMul_3:product:0lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_4Mulzeros:output:0 lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_5Mulzeros:output:0 lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_6Mulzeros:output:0 lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_7Mulzeros:output:0 lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_cell_22/ReadVariableOpReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      г
lstm_cell_22/strided_sliceStridedSlice#lstm_cell_22/ReadVariableOp:value:0)lstm_cell_22/strided_slice/stack:output:0+lstm_cell_22/strided_slice/stack_1:output:0+lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskј
lstm_cell_22/MatMul_4MatMullstm_cell_22/mul_4:z:0#lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         І
lstm_cell_22/addAddV2lstm_cell_22/BiasAdd:output:0lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         g
lstm_cell_22/SigmoidSigmoidlstm_cell_22/add:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_1ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_1StridedSlice%lstm_cell_22/ReadVariableOp_1:value:0+lstm_cell_22/strided_slice_1/stack:output:0-lstm_cell_22/strided_slice_1/stack_1:output:0-lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_5MatMullstm_cell_22/mul_5:z:0%lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_1AddV2lstm_cell_22/BiasAdd_1:output:0lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         y
lstm_cell_22/mul_8Mullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_2ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_2StridedSlice%lstm_cell_22/ReadVariableOp_2:value:0+lstm_cell_22/strided_slice_2/stack:output:0-lstm_cell_22/strided_slice_2/stack_1:output:0-lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_6MatMullstm_cell_22/mul_6:z:0%lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_2AddV2lstm_cell_22/BiasAdd_2:output:0lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell_22/mul_9Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         }
lstm_cell_22/add_3AddV2lstm_cell_22/mul_8:z:0lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_3ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_3StridedSlice%lstm_cell_22/ReadVariableOp_3:value:0+lstm_cell_22/strided_slice_3/stack:output:0-lstm_cell_22/strided_slice_3/stack_1:output:0-lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_7MatMullstm_cell_22/mul_7:z:0%lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_4AddV2lstm_cell_22/BiasAdd_3:output:0lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_3Sigmoidlstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_4Sigmoidlstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mul_10Mullstm_cell_22/Sigmoid_3:y:0lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_22_split_readvariableop_resource,lstm_cell_22_split_1_readvariableop_resource$lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_444031*
condR
while_cond_444030*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp^lstm_cell_22/ReadVariableOp^lstm_cell_22/ReadVariableOp_1^lstm_cell_22/ReadVariableOp_2^lstm_cell_22/ReadVariableOp_3"^lstm_cell_22/split/ReadVariableOp$^lstm_cell_22/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2:
lstm_cell_22/ReadVariableOplstm_cell_22/ReadVariableOp2>
lstm_cell_22/ReadVariableOp_1lstm_cell_22/ReadVariableOp_12>
lstm_cell_22/ReadVariableOp_2lstm_cell_22/ReadVariableOp_22>
lstm_cell_22/ReadVariableOp_3lstm_cell_22/ReadVariableOp_32F
!lstm_cell_22/split/ReadVariableOp!lstm_cell_22/split/ReadVariableOp2J
#lstm_cell_22/split_1/ReadVariableOp#lstm_cell_22/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ц
Џ
I__inference_sequential_21_layer_call_and_return_conditional_losses_441560

inputs!
dense_66_441272:
dense_66_441274: 
lstm_22_441520:
lstm_22_441522: 
lstm_22_441524:!
dense_67_441538:
dense_67_441540:!
dense_68_441554:
dense_68_441556:
identityѕб dense_66/StatefulPartitionedCallб dense_67/StatefulPartitionedCallб dense_68/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallЗ
 dense_66/StatefulPartitionedCallStatefulPartitionedCallinputsdense_66_441272dense_66_441274*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_66_layer_call_and_return_conditional_losses_441271А
lstm_22/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0lstm_22_441520lstm_22_441522lstm_22_441524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_441519њ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_67_441538dense_67_441540*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_67_layer_call_and_return_conditional_losses_441537Њ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_441554dense_68_441556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_441553x
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Л
NoOpNoOp!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
н─
ъ	
while_body_443417
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_22_split_readvariableop_resource_0:B
4while_lstm_cell_22_split_1_readvariableop_resource_0:>
,while_lstm_cell_22_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_22_split_readvariableop_resource:@
2while_lstm_cell_22_split_1_readvariableop_resource:<
*while_lstm_cell_22_readvariableop_resource:ѕб!while/lstm_cell_22/ReadVariableOpб#while/lstm_cell_22/ReadVariableOp_1б#while/lstm_cell_22/ReadVariableOp_2б#while/lstm_cell_22/ReadVariableOp_3б'while/lstm_cell_22/split/ReadVariableOpб)while/lstm_cell_22/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ѓ
"while/lstm_cell_22/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?░
while/lstm_cell_22/ones_likeFill+while/lstm_cell_22/ones_like/Shape:output:0+while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         e
 while/lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Е
while/lstm_cell_22/dropout/MulMul%while/lstm_cell_22/ones_like:output:0)while/lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         u
 while/lstm_cell_22/dropout/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:▓
7while/lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0n
)while/lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=▀
'while/lstm_cell_22/dropout/GreaterEqualGreaterEqual@while/lstm_cell_22/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ћ
while/lstm_cell_22/dropout/CastCast+while/lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         б
 while/lstm_cell_22/dropout/Mul_1Mul"while/lstm_cell_22/dropout/Mul:z:0#while/lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_1/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_1/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_1/CastCast-while/lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_1/Mul_1Mul$while/lstm_cell_22/dropout_1/Mul:z:0%while/lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_2/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_2/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_2/CastCast-while/lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_2/Mul_1Mul$while/lstm_cell_22/dropout_2/Mul:z:0%while/lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_3/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_3/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_3/CastCast-while/lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_3/Mul_1Mul$while/lstm_cell_22/dropout_3/Mul:z:0%while/lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         g
$while/lstm_cell_22/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
while/lstm_cell_22/ones_like_1Fill-while/lstm_cell_22/ones_like_1/Shape:output:0-while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_4/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_4/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_4/CastCast-while/lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_4/Mul_1Mul$while/lstm_cell_22/dropout_4/Mul:z:0%while/lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_5/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_5/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_5/CastCast-while/lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_5/Mul_1Mul$while/lstm_cell_22/dropout_5/Mul:z:0%while/lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_6/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_6/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_6/CastCast-while/lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_6/Mul_1Mul$while/lstm_cell_22/dropout_6/Mul:z:0%while/lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_7/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_7/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_7/CastCast-while/lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_7/Mul_1Mul$while/lstm_cell_22/dropout_7/Mul:z:0%while/lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         Д
while/lstm_cell_22/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :џ
'while/lstm_cell_22/split/ReadVariableOpReadVariableOp2while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0О
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitћ
while/lstm_cell_22/MatMulMatMulwhile/lstm_cell_22/mul:z:0!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_1MatMulwhile/lstm_cell_22/mul_1:z:0!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_2MatMulwhile/lstm_cell_22/mul_2:z:0!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_3MatMulwhile/lstm_cell_22/mul_3:z:0!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         f
$while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : џ
)while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0═
while/lstm_cell_22/split_1Split-while/lstm_cell_22/split_1/split_dim:output:01while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitА
while/lstm_cell_22/BiasAddBiasAdd#while/lstm_cell_22/MatMul:product:0#while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_1BiasAdd%while/lstm_cell_22/MatMul_1:product:0#while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_2BiasAdd%while/lstm_cell_22/MatMul_2:product:0#while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_3BiasAdd%while/lstm_cell_22/MatMul_3:product:0#while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_4Mulwhile_placeholder_2&while/lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_5Mulwhile_placeholder_2&while/lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_6Mulwhile_placeholder_2&while/lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_7Mulwhile_placeholder_2&while/lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         ј
!while/lstm_cell_22/ReadVariableOpReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_22/strided_sliceStridedSlice)while/lstm_cell_22/ReadVariableOp:value:0/while/lstm_cell_22/strided_slice/stack:output:01while/lstm_cell_22/strided_slice/stack_1:output:01while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskа
while/lstm_cell_22/MatMul_4MatMulwhile/lstm_cell_22/mul_4:z:0)while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/BiasAdd:output:0%while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         s
while/lstm_cell_22/SigmoidSigmoidwhile/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_1ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_1StridedSlice+while/lstm_cell_22/ReadVariableOp_1:value:01while/lstm_cell_22/strided_slice_1/stack:output:03while/lstm_cell_22/strided_slice_1/stack_1:output:03while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_5MatMulwhile/lstm_cell_22/mul_5:z:0+while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_1AddV2%while/lstm_cell_22/BiasAdd_1:output:0%while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_1Sigmoidwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ѕ
while/lstm_cell_22/mul_8Mul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_2ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_2StridedSlice+while/lstm_cell_22/ReadVariableOp_2:value:01while/lstm_cell_22/strided_slice_2/stack:output:03while/lstm_cell_22/strided_slice_2/stack_1:output:03while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_6MatMulwhile/lstm_cell_22/mul_6:z:0+while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_2AddV2%while/lstm_cell_22/BiasAdd_2:output:0%while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_2Sigmoidwhile/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell_22/mul_9Mulwhile/lstm_cell_22/Sigmoid:y:0 while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/add_3AddV2while/lstm_cell_22/mul_8:z:0while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_3ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_3StridedSlice+while/lstm_cell_22/ReadVariableOp_3:value:01while/lstm_cell_22/strided_slice_3/stack:output:03while/lstm_cell_22/strided_slice_3/stack_1:output:03while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_7MatMulwhile/lstm_cell_22/mul_7:z:0+while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_4AddV2%while/lstm_cell_22/BiasAdd_3:output:0%while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_3Sigmoidwhile/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_4Sigmoidwhile/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ќ
while/lstm_cell_22/mul_10Mul while/lstm_cell_22/Sigmoid_3:y:0 while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         к
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмz
while/Identity_4Identitywhile/lstm_cell_22/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         И

while/NoOpNoOp"^while/lstm_cell_22/ReadVariableOp$^while/lstm_cell_22/ReadVariableOp_1$^while/lstm_cell_22/ReadVariableOp_2$^while/lstm_cell_22/ReadVariableOp_3(^while/lstm_cell_22/split/ReadVariableOp*^while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_22_readvariableop_resource,while_lstm_cell_22_readvariableop_resource_0"j
2while_lstm_cell_22_split_1_readvariableop_resource4while_lstm_cell_22_split_1_readvariableop_resource_0"f
0while_lstm_cell_22_split_readvariableop_resource2while_lstm_cell_22_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2F
!while/lstm_cell_22/ReadVariableOp!while/lstm_cell_22/ReadVariableOp2J
#while/lstm_cell_22/ReadVariableOp_1#while/lstm_cell_22/ReadVariableOp_12J
#while/lstm_cell_22/ReadVariableOp_2#while/lstm_cell_22/ReadVariableOp_22J
#while/lstm_cell_22/ReadVariableOp_3#while/lstm_cell_22/ReadVariableOp_32R
'while/lstm_cell_22/split/ReadVariableOp'while/lstm_cell_22/split/ReadVariableOp2V
)while/lstm_cell_22/split_1/ReadVariableOp)while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
═~
е
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_444529

inputs
states_0
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         R
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:         O
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:ї
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=д
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         o
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         i
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:љ
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:љ
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:         Q
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:љ
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:         I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:         T
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:         S
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:         S
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:         S
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:         T
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:         S
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:љ
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=г
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         s
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         o
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:         W
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         [
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0ъ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         _
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         _
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         _
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         ]
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         ]
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         ]
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         ]
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:         Z
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:         V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         Q
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:         ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         [

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
х
├
while_cond_441155
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_441155___redundant_placeholder04
0while_while_cond_441155___redundant_placeholder14
0while_while_cond_441155___redundant_placeholder24
0while_while_cond_441155___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
╝
Б
I__inference_sequential_21_layer_call_and_return_conditional_losses_442151
dense_66_input!
dense_66_442128:
dense_66_442130: 
lstm_22_442133:
lstm_22_442135: 
lstm_22_442137:!
dense_67_442140:
dense_67_442142:!
dense_68_442145:
dense_68_442147:
identityѕб dense_66/StatefulPartitionedCallб dense_67/StatefulPartitionedCallб dense_68/StatefulPartitionedCallбlstm_22/StatefulPartitionedCallЧ
 dense_66/StatefulPartitionedCallStatefulPartitionedCalldense_66_inputdense_66_442128dense_66_442130*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_66_layer_call_and_return_conditional_losses_441271А
lstm_22/StatefulPartitionedCallStatefulPartitionedCall)dense_66/StatefulPartitionedCall:output:0lstm_22_442133lstm_22_442135lstm_22_442137*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_441985њ
 dense_67/StatefulPartitionedCallStatefulPartitionedCall(lstm_22/StatefulPartitionedCall:output:0dense_67_442140dense_67_442142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_67_layer_call_and_return_conditional_losses_441537Њ
 dense_68/StatefulPartitionedCallStatefulPartitionedCall)dense_67/StatefulPartitionedCall:output:0dense_68_442145dense_68_442147*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_68_layer_call_and_return_conditional_losses_441553x
IdentityIdentity)dense_68/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Л
NoOpNoOp!^dense_66/StatefulPartitionedCall!^dense_67/StatefulPartitionedCall!^dense_68/StatefulPartitionedCall ^lstm_22/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 2D
 dense_66/StatefulPartitionedCall dense_66/StatefulPartitionedCall2D
 dense_67/StatefulPartitionedCall dense_67/StatefulPartitionedCall2D
 dense_68/StatefulPartitionedCall dense_68/StatefulPartitionedCall2B
lstm_22/StatefulPartitionedCalllstm_22/StatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_namedense_66_input
ћ

Р
.__inference_sequential_21_layer_call_fn_442099
dense_66_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityѕбStatefulPartitionedCall┴
StatefulPartitionedCallStatefulPartitionedCalldense_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_442055o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_namedense_66_input
У
з
-__inference_lstm_cell_22_layer_call_fn_444284

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2ѕбStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_440837o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         q

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:         q

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
▀Ђ
С
C__inference_lstm_22_layer_call_and_return_conditional_losses_443858

inputs<
*lstm_cell_22_split_readvariableop_resource::
,lstm_cell_22_split_1_readvariableop_resource:6
$lstm_cell_22_readvariableop_resource:
identityѕбlstm_cell_22/ReadVariableOpбlstm_cell_22/ReadVariableOp_1бlstm_cell_22/ReadVariableOp_2бlstm_cell_22/ReadVariableOp_3б!lstm_cell_22/split/ReadVariableOpб#lstm_cell_22/split_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskd
lstm_cell_22/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ъ
lstm_cell_22/ones_likeFill%lstm_cell_22/ones_like/Shape:output:0%lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         \
lstm_cell_22/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ц
lstm_cell_22/ones_like_1Fill'lstm_cell_22/ones_like_1/Shape:output:0'lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mulMulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_2Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_3Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
!lstm_cell_22/split/ReadVariableOpReadVariableOp*lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0┼
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0)lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitѓ
lstm_cell_22/MatMulMatMullstm_cell_22/mul:z:0lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_1MatMullstm_cell_22/mul_1:z:0lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_2MatMullstm_cell_22/mul_2:z:0lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_3MatMullstm_cell_22/mul_3:z:0lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         `
lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ї
#lstm_cell_22/split_1/ReadVariableOpReadVariableOp,lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0╗
lstm_cell_22/split_1Split'lstm_cell_22/split_1/split_dim:output:0+lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЈ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/MatMul:product:0lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_1BiasAddlstm_cell_22/MatMul_1:product:0lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_2BiasAddlstm_cell_22/MatMul_2:product:0lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_3BiasAddlstm_cell_22/MatMul_3:product:0lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_4Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_5Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_6Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_7Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ђ
lstm_cell_22/ReadVariableOpReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      г
lstm_cell_22/strided_sliceStridedSlice#lstm_cell_22/ReadVariableOp:value:0)lstm_cell_22/strided_slice/stack:output:0+lstm_cell_22/strided_slice/stack_1:output:0+lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskј
lstm_cell_22/MatMul_4MatMullstm_cell_22/mul_4:z:0#lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         І
lstm_cell_22/addAddV2lstm_cell_22/BiasAdd:output:0lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         g
lstm_cell_22/SigmoidSigmoidlstm_cell_22/add:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_1ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_1StridedSlice%lstm_cell_22/ReadVariableOp_1:value:0+lstm_cell_22/strided_slice_1/stack:output:0-lstm_cell_22/strided_slice_1/stack_1:output:0-lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_5MatMullstm_cell_22/mul_5:z:0%lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_1AddV2lstm_cell_22/BiasAdd_1:output:0lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         y
lstm_cell_22/mul_8Mullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_2ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_2StridedSlice%lstm_cell_22/ReadVariableOp_2:value:0+lstm_cell_22/strided_slice_2/stack:output:0-lstm_cell_22/strided_slice_2/stack_1:output:0-lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_6MatMullstm_cell_22/mul_6:z:0%lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_2AddV2lstm_cell_22/BiasAdd_2:output:0lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell_22/mul_9Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         }
lstm_cell_22/add_3AddV2lstm_cell_22/mul_8:z:0lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_3ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_3StridedSlice%lstm_cell_22/ReadVariableOp_3:value:0+lstm_cell_22/strided_slice_3/stack:output:0-lstm_cell_22/strided_slice_3/stack_1:output:0-lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_7MatMullstm_cell_22/mul_7:z:0%lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_4AddV2lstm_cell_22/BiasAdd_3:output:0lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_3Sigmoidlstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_4Sigmoidlstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mul_10Mullstm_cell_22/Sigmoid_3:y:0lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_22_split_readvariableop_resource,lstm_cell_22_split_1_readvariableop_resource$lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_443724*
condR
while_cond_443723*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp^lstm_cell_22/ReadVariableOp^lstm_cell_22/ReadVariableOp_1^lstm_cell_22/ReadVariableOp_2^lstm_cell_22/ReadVariableOp_3"^lstm_cell_22/split/ReadVariableOp$^lstm_cell_22/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2:
lstm_cell_22/ReadVariableOplstm_cell_22/ReadVariableOp2>
lstm_cell_22/ReadVariableOp_1lstm_cell_22/ReadVariableOp_12>
lstm_cell_22/ReadVariableOp_2lstm_cell_22/ReadVariableOp_22>
lstm_cell_22/ReadVariableOp_3lstm_cell_22/ReadVariableOp_32F
!lstm_cell_22/split/ReadVariableOp!lstm_cell_22/split/ReadVariableOp2J
#lstm_cell_22/split_1/ReadVariableOp#lstm_cell_22/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Ч	
┌
.__inference_sequential_21_layer_call_fn_442180

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_441560o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
Р	
п
$__inference_signature_wrapper_442918
dense_66_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityѕбStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCalldense_66_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8ѓ **
f%R#
!__inference__wrapped_model_440720o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:         
(
_user_specified_namedense_66_input
К	
ш
D__inference_dense_67_layer_call_and_return_conditional_losses_441537

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ѓ└
Р
I__inference_sequential_21_layer_call_and_return_conditional_losses_442484

inputs<
*dense_66_tensordot_readvariableop_resource:6
(dense_66_biasadd_readvariableop_resource:D
2lstm_22_lstm_cell_22_split_readvariableop_resource:B
4lstm_22_lstm_cell_22_split_1_readvariableop_resource:>
,lstm_22_lstm_cell_22_readvariableop_resource:9
'dense_67_matmul_readvariableop_resource:6
(dense_67_biasadd_readvariableop_resource:9
'dense_68_matmul_readvariableop_resource:6
(dense_68_biasadd_readvariableop_resource:
identityѕбdense_66/BiasAdd/ReadVariableOpб!dense_66/Tensordot/ReadVariableOpбdense_67/BiasAdd/ReadVariableOpбdense_67/MatMul/ReadVariableOpбdense_68/BiasAdd/ReadVariableOpбdense_68/MatMul/ReadVariableOpб#lstm_22/lstm_cell_22/ReadVariableOpб%lstm_22/lstm_cell_22/ReadVariableOp_1б%lstm_22/lstm_cell_22/ReadVariableOp_2б%lstm_22/lstm_cell_22/ReadVariableOp_3б)lstm_22/lstm_cell_22/split/ReadVariableOpб+lstm_22/lstm_cell_22/split_1/ReadVariableOpбlstm_22/whileї
!dense_66/Tensordot/ReadVariableOpReadVariableOp*dense_66_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_66/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_66/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_66/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_66/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
dense_66/Tensordot/GatherV2GatherV2!dense_66/Tensordot/Shape:output:0 dense_66/Tensordot/free:output:0)dense_66/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_66/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : с
dense_66/Tensordot/GatherV2_1GatherV2!dense_66/Tensordot/Shape:output:0 dense_66/Tensordot/axes:output:0+dense_66/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_66/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ѕ
dense_66/Tensordot/ProdProd$dense_66/Tensordot/GatherV2:output:0!dense_66/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_66/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ј
dense_66/Tensordot/Prod_1Prod&dense_66/Tensordot/GatherV2_1:output:0#dense_66/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_66/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : └
dense_66/Tensordot/concatConcatV2 dense_66/Tensordot/free:output:0 dense_66/Tensordot/axes:output:0'dense_66/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ћ
dense_66/Tensordot/stackPack dense_66/Tensordot/Prod:output:0"dense_66/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:І
dense_66/Tensordot/transpose	Transposeinputs"dense_66/Tensordot/concat:output:0*
T0*+
_output_shapes
:         Ц
dense_66/Tensordot/ReshapeReshape dense_66/Tensordot/transpose:y:0!dense_66/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ц
dense_66/Tensordot/MatMulMatMul#dense_66/Tensordot/Reshape:output:0)dense_66/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_66/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_66/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
dense_66/Tensordot/concat_1ConcatV2$dense_66/Tensordot/GatherV2:output:0#dense_66/Tensordot/Const_2:output:0)dense_66/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ъ
dense_66/TensordotReshape#dense_66/Tensordot/MatMul:product:0$dense_66/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         ё
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
dense_66/BiasAddBiasAdddense_66/Tensordot:output:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         V
lstm_22/ShapeShapedense_66/BiasAdd:output:0*
T0*
_output_shapes
:e
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
lstm_22/transpose	Transposedense_66/BiasAdd:output:0lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:g
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskt
$lstm_22/lstm_cell_22/ones_like/ShapeShape lstm_22/strided_slice_2:output:0*
T0*
_output_shapes
:i
$lstm_22/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
lstm_22/lstm_cell_22/ones_likeFill-lstm_22/lstm_cell_22/ones_like/Shape:output:0-lstm_22/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         l
&lstm_22/lstm_cell_22/ones_like_1/ShapeShapelstm_22/zeros:output:0*
T0*
_output_shapes
:k
&lstm_22/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?╝
 lstm_22/lstm_cell_22/ones_like_1Fill/lstm_22/lstm_cell_22/ones_like_1/Shape:output:0/lstm_22/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         ю
lstm_22/lstm_cell_22/mulMul lstm_22/strided_slice_2:output:0'lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/mul_1Mul lstm_22/strided_slice_2:output:0'lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/mul_2Mul lstm_22/strided_slice_2:output:0'lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/mul_3Mul lstm_22/strided_slice_2:output:0'lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         f
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ю
)lstm_22/lstm_cell_22/split/ReadVariableOpReadVariableOp2lstm_22_lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:01lstm_22/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitџ
lstm_22/lstm_cell_22/MatMulMatMullstm_22/lstm_cell_22/mul:z:0#lstm_22/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/lstm_cell_22/mul_1:z:0#lstm_22/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/MatMul_2MatMullstm_22/lstm_cell_22/mul_2:z:0#lstm_22/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/MatMul_3MatMullstm_22/lstm_cell_22/mul_3:z:0#lstm_22/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         h
&lstm_22/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ю
+lstm_22/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0М
lstm_22/lstm_cell_22/split_1Split/lstm_22/lstm_cell_22/split_1/split_dim:output:03lstm_22/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitД
lstm_22/lstm_cell_22/BiasAddBiasAdd%lstm_22/lstm_cell_22/MatMul:product:0%lstm_22/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ф
lstm_22/lstm_cell_22/BiasAdd_1BiasAdd'lstm_22/lstm_cell_22/MatMul_1:product:0%lstm_22/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ф
lstm_22/lstm_cell_22/BiasAdd_2BiasAdd'lstm_22/lstm_cell_22/MatMul_2:product:0%lstm_22/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ф
lstm_22/lstm_cell_22/BiasAdd_3BiasAdd'lstm_22/lstm_cell_22/MatMul_3:product:0%lstm_22/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         ќ
lstm_22/lstm_cell_22/mul_4Mullstm_22/zeros:output:0)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ќ
lstm_22/lstm_cell_22/mul_5Mullstm_22/zeros:output:0)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ќ
lstm_22/lstm_cell_22/mul_6Mullstm_22/zeros:output:0)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ќ
lstm_22/lstm_cell_22/mul_7Mullstm_22/zeros:output:0)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         љ
#lstm_22/lstm_cell_22/ReadVariableOpReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0y
(lstm_22/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_22/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_22/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"lstm_22/lstm_cell_22/strided_sliceStridedSlice+lstm_22/lstm_cell_22/ReadVariableOp:value:01lstm_22/lstm_cell_22/strided_slice/stack:output:03lstm_22/lstm_cell_22/strided_slice/stack_1:output:03lstm_22/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskд
lstm_22/lstm_cell_22/MatMul_4MatMullstm_22/lstm_cell_22/mul_4:z:0+lstm_22/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Б
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/BiasAdd:output:0'lstm_22/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         w
lstm_22/lstm_cell_22/SigmoidSigmoidlstm_22/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         њ
%lstm_22/lstm_cell_22/ReadVariableOp_1ReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_22/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      я
$lstm_22/lstm_cell_22/strided_slice_1StridedSlice-lstm_22/lstm_cell_22/ReadVariableOp_1:value:03lstm_22/lstm_cell_22/strided_slice_1/stack:output:05lstm_22/lstm_cell_22/strided_slice_1/stack_1:output:05lstm_22/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskе
lstm_22/lstm_cell_22/MatMul_5MatMullstm_22/lstm_cell_22/mul_5:z:0-lstm_22/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Д
lstm_22/lstm_cell_22/add_1AddV2'lstm_22/lstm_cell_22/BiasAdd_1:output:0'lstm_22/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_1Sigmoidlstm_22/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         Љ
lstm_22/lstm_cell_22/mul_8Mul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*'
_output_shapes
:         њ
%lstm_22/lstm_cell_22/ReadVariableOp_2ReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_22/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      я
$lstm_22/lstm_cell_22/strided_slice_2StridedSlice-lstm_22/lstm_cell_22/ReadVariableOp_2:value:03lstm_22/lstm_cell_22/strided_slice_2/stack:output:05lstm_22/lstm_cell_22/strided_slice_2/stack_1:output:05lstm_22/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskе
lstm_22/lstm_cell_22/MatMul_6MatMullstm_22/lstm_cell_22/mul_6:z:0-lstm_22/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Д
lstm_22/lstm_cell_22/add_2AddV2'lstm_22/lstm_cell_22/BiasAdd_2:output:0'lstm_22/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_2Sigmoidlstm_22/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ў
lstm_22/lstm_cell_22/mul_9Mul lstm_22/lstm_cell_22/Sigmoid:y:0"lstm_22/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ћ
lstm_22/lstm_cell_22/add_3AddV2lstm_22/lstm_cell_22/mul_8:z:0lstm_22/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         њ
%lstm_22/lstm_cell_22/ReadVariableOp_3ReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_22/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_22/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      я
$lstm_22/lstm_cell_22/strided_slice_3StridedSlice-lstm_22/lstm_cell_22/ReadVariableOp_3:value:03lstm_22/lstm_cell_22/strided_slice_3/stack:output:05lstm_22/lstm_cell_22/strided_slice_3/stack_1:output:05lstm_22/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskе
lstm_22/lstm_cell_22/MatMul_7MatMullstm_22/lstm_cell_22/mul_7:z:0-lstm_22/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Д
lstm_22/lstm_cell_22/add_4AddV2'lstm_22/lstm_cell_22/BiasAdd_3:output:0'lstm_22/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_3Sigmoidlstm_22/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_4Sigmoidlstm_22/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ю
lstm_22/lstm_cell_22/mul_10Mul"lstm_22/lstm_cell_22/Sigmoid_3:y:0"lstm_22/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         v
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_22_lstm_cell_22_split_readvariableop_resource4lstm_22_lstm_cell_22_split_1_readvariableop_resource,lstm_22_lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_22_while_body_442338*%
condR
lstm_22_while_cond_442337*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    є
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ћ
dense_67/MatMulMatMul lstm_22/strided_slice_3:output:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         є
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ј
dense_68/MatMulMatMuldense_67/BiasAdd:output:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_68/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         џ
NoOpNoOp ^dense_66/BiasAdd/ReadVariableOp"^dense_66/Tensordot/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp$^lstm_22/lstm_cell_22/ReadVariableOp&^lstm_22/lstm_cell_22/ReadVariableOp_1&^lstm_22/lstm_cell_22/ReadVariableOp_2&^lstm_22/lstm_cell_22/ReadVariableOp_3*^lstm_22/lstm_cell_22/split/ReadVariableOp,^lstm_22/lstm_cell_22/split_1/ReadVariableOp^lstm_22/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2F
!dense_66/Tensordot/ReadVariableOp!dense_66/Tensordot/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2J
#lstm_22/lstm_cell_22/ReadVariableOp#lstm_22/lstm_cell_22/ReadVariableOp2N
%lstm_22/lstm_cell_22/ReadVariableOp_1%lstm_22/lstm_cell_22/ReadVariableOp_12N
%lstm_22/lstm_cell_22/ReadVariableOp_2%lstm_22/lstm_cell_22/ReadVariableOp_22N
%lstm_22/lstm_cell_22/ReadVariableOp_3%lstm_22/lstm_cell_22/ReadVariableOp_32V
)lstm_22/lstm_cell_22/split/ReadVariableOp)lstm_22/lstm_cell_22/split/ReadVariableOp2Z
+lstm_22/lstm_cell_22/split_1/ReadVariableOp+lstm_22/lstm_cell_22/split_1/ReadVariableOp2
lstm_22/whilelstm_22/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ИЖ
Ш

!__inference__wrapped_model_440720
dense_66_inputJ
8sequential_21_dense_66_tensordot_readvariableop_resource:D
6sequential_21_dense_66_biasadd_readvariableop_resource:R
@sequential_21_lstm_22_lstm_cell_22_split_readvariableop_resource:P
Bsequential_21_lstm_22_lstm_cell_22_split_1_readvariableop_resource:L
:sequential_21_lstm_22_lstm_cell_22_readvariableop_resource:G
5sequential_21_dense_67_matmul_readvariableop_resource:D
6sequential_21_dense_67_biasadd_readvariableop_resource:G
5sequential_21_dense_68_matmul_readvariableop_resource:D
6sequential_21_dense_68_biasadd_readvariableop_resource:
identityѕб-sequential_21/dense_66/BiasAdd/ReadVariableOpб/sequential_21/dense_66/Tensordot/ReadVariableOpб-sequential_21/dense_67/BiasAdd/ReadVariableOpб,sequential_21/dense_67/MatMul/ReadVariableOpб-sequential_21/dense_68/BiasAdd/ReadVariableOpб,sequential_21/dense_68/MatMul/ReadVariableOpб1sequential_21/lstm_22/lstm_cell_22/ReadVariableOpб3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_1б3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_2б3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_3б7sequential_21/lstm_22/lstm_cell_22/split/ReadVariableOpб9sequential_21/lstm_22/lstm_cell_22/split_1/ReadVariableOpбsequential_21/lstm_22/whileе
/sequential_21/dense_66/Tensordot/ReadVariableOpReadVariableOp8sequential_21_dense_66_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0o
%sequential_21/dense_66/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_21/dense_66/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       d
&sequential_21/dense_66/Tensordot/ShapeShapedense_66_input*
T0*
_output_shapes
:p
.sequential_21/dense_66/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Ќ
)sequential_21/dense_66/Tensordot/GatherV2GatherV2/sequential_21/dense_66/Tensordot/Shape:output:0.sequential_21/dense_66/Tensordot/free:output:07sequential_21/dense_66/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_21/dense_66/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Џ
+sequential_21/dense_66/Tensordot/GatherV2_1GatherV2/sequential_21/dense_66/Tensordot/Shape:output:0.sequential_21/dense_66/Tensordot/axes:output:09sequential_21/dense_66/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_21/dense_66/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: │
%sequential_21/dense_66/Tensordot/ProdProd2sequential_21/dense_66/Tensordot/GatherV2:output:0/sequential_21/dense_66/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_21/dense_66/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ╣
'sequential_21/dense_66/Tensordot/Prod_1Prod4sequential_21/dense_66/Tensordot/GatherV2_1:output:01sequential_21/dense_66/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_21/dense_66/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Э
'sequential_21/dense_66/Tensordot/concatConcatV2.sequential_21/dense_66/Tensordot/free:output:0.sequential_21/dense_66/Tensordot/axes:output:05sequential_21/dense_66/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Й
&sequential_21/dense_66/Tensordot/stackPack.sequential_21/dense_66/Tensordot/Prod:output:00sequential_21/dense_66/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:»
*sequential_21/dense_66/Tensordot/transpose	Transposedense_66_input0sequential_21/dense_66/Tensordot/concat:output:0*
T0*+
_output_shapes
:         ¤
(sequential_21/dense_66/Tensordot/ReshapeReshape.sequential_21/dense_66/Tensordot/transpose:y:0/sequential_21/dense_66/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ¤
'sequential_21/dense_66/Tensordot/MatMulMatMul1sequential_21/dense_66/Tensordot/Reshape:output:07sequential_21/dense_66/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
(sequential_21/dense_66/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:p
.sequential_21/dense_66/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ѓ
)sequential_21/dense_66/Tensordot/concat_1ConcatV22sequential_21/dense_66/Tensordot/GatherV2:output:01sequential_21/dense_66/Tensordot/Const_2:output:07sequential_21/dense_66/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╚
 sequential_21/dense_66/TensordotReshape1sequential_21/dense_66/Tensordot/MatMul:product:02sequential_21/dense_66/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         а
-sequential_21/dense_66/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_66_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0┴
sequential_21/dense_66/BiasAddBiasAdd)sequential_21/dense_66/Tensordot:output:05sequential_21/dense_66/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         r
sequential_21/lstm_22/ShapeShape'sequential_21/dense_66/BiasAdd:output:0*
T0*
_output_shapes
:s
)sequential_21/lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_21/lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_21/lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:┐
#sequential_21/lstm_22/strided_sliceStridedSlice$sequential_21/lstm_22/Shape:output:02sequential_21/lstm_22/strided_slice/stack:output:04sequential_21/lstm_22/strided_slice/stack_1:output:04sequential_21/lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_21/lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :х
"sequential_21/lstm_22/zeros/packedPack,sequential_21/lstm_22/strided_slice:output:0-sequential_21/lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_21/lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    «
sequential_21/lstm_22/zerosFill+sequential_21/lstm_22/zeros/packed:output:0*sequential_21/lstm_22/zeros/Const:output:0*
T0*'
_output_shapes
:         h
&sequential_21/lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :╣
$sequential_21/lstm_22/zeros_1/packedPack,sequential_21/lstm_22/strided_slice:output:0/sequential_21/lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_21/lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ┤
sequential_21/lstm_22/zeros_1Fill-sequential_21/lstm_22/zeros_1/packed:output:0,sequential_21/lstm_22/zeros_1/Const:output:0*
T0*'
_output_shapes
:         y
$sequential_21/lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          ║
sequential_21/lstm_22/transpose	Transpose'sequential_21/dense_66/BiasAdd:output:0-sequential_21/lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         p
sequential_21/lstm_22/Shape_1Shape#sequential_21/lstm_22/transpose:y:0*
T0*
_output_shapes
:u
+sequential_21/lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_21/lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_21/lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:╔
%sequential_21/lstm_22/strided_slice_1StridedSlice&sequential_21/lstm_22/Shape_1:output:04sequential_21/lstm_22/strided_slice_1/stack:output:06sequential_21/lstm_22/strided_slice_1/stack_1:output:06sequential_21/lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_21/lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         Ш
#sequential_21/lstm_22/TensorArrayV2TensorListReserve:sequential_21/lstm_22/TensorArrayV2/element_shape:output:0.sequential_21/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмю
Ksequential_21/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       б
=sequential_21/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_21/lstm_22/transpose:y:0Tsequential_21/lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмu
+sequential_21/lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_21/lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_21/lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:О
%sequential_21/lstm_22/strided_slice_2StridedSlice#sequential_21/lstm_22/transpose:y:04sequential_21/lstm_22/strided_slice_2/stack:output:06sequential_21/lstm_22/strided_slice_2/stack_1:output:06sequential_21/lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskљ
2sequential_21/lstm_22/lstm_cell_22/ones_like/ShapeShape.sequential_21/lstm_22/strided_slice_2:output:0*
T0*
_output_shapes
:w
2sequential_21/lstm_22/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Я
,sequential_21/lstm_22/lstm_cell_22/ones_likeFill;sequential_21/lstm_22/lstm_cell_22/ones_like/Shape:output:0;sequential_21/lstm_22/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         ѕ
4sequential_21/lstm_22/lstm_cell_22/ones_like_1/ShapeShape$sequential_21/lstm_22/zeros:output:0*
T0*
_output_shapes
:y
4sequential_21/lstm_22/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Т
.sequential_21/lstm_22/lstm_cell_22/ones_like_1Fill=sequential_21/lstm_22/lstm_cell_22/ones_like_1/Shape:output:0=sequential_21/lstm_22/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         к
&sequential_21/lstm_22/lstm_cell_22/mulMul.sequential_21/lstm_22/strided_slice_2:output:05sequential_21/lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ╚
(sequential_21/lstm_22/lstm_cell_22/mul_1Mul.sequential_21/lstm_22/strided_slice_2:output:05sequential_21/lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ╚
(sequential_21/lstm_22/lstm_cell_22/mul_2Mul.sequential_21/lstm_22/strided_slice_2:output:05sequential_21/lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ╚
(sequential_21/lstm_22/lstm_cell_22/mul_3Mul.sequential_21/lstm_22/strided_slice_2:output:05sequential_21/lstm_22/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         t
2sequential_21/lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :И
7sequential_21/lstm_22/lstm_cell_22/split/ReadVariableOpReadVariableOp@sequential_21_lstm_22_lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0Є
(sequential_21/lstm_22/lstm_cell_22/splitSplit;sequential_21/lstm_22/lstm_cell_22/split/split_dim:output:0?sequential_21/lstm_22/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split─
)sequential_21/lstm_22/lstm_cell_22/MatMulMatMul*sequential_21/lstm_22/lstm_cell_22/mul:z:01sequential_21/lstm_22/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ╚
+sequential_21/lstm_22/lstm_cell_22/MatMul_1MatMul,sequential_21/lstm_22/lstm_cell_22/mul_1:z:01sequential_21/lstm_22/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ╚
+sequential_21/lstm_22/lstm_cell_22/MatMul_2MatMul,sequential_21/lstm_22/lstm_cell_22/mul_2:z:01sequential_21/lstm_22/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ╚
+sequential_21/lstm_22/lstm_cell_22/MatMul_3MatMul,sequential_21/lstm_22/lstm_cell_22/mul_3:z:01sequential_21/lstm_22/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         v
4sequential_21/lstm_22/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : И
9sequential_21/lstm_22/lstm_cell_22/split_1/ReadVariableOpReadVariableOpBsequential_21_lstm_22_lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0§
*sequential_21/lstm_22/lstm_cell_22/split_1Split=sequential_21/lstm_22/lstm_cell_22/split_1/split_dim:output:0Asequential_21/lstm_22/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЛ
*sequential_21/lstm_22/lstm_cell_22/BiasAddBiasAdd3sequential_21/lstm_22/lstm_cell_22/MatMul:product:03sequential_21/lstm_22/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Н
,sequential_21/lstm_22/lstm_cell_22/BiasAdd_1BiasAdd5sequential_21/lstm_22/lstm_cell_22/MatMul_1:product:03sequential_21/lstm_22/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Н
,sequential_21/lstm_22/lstm_cell_22/BiasAdd_2BiasAdd5sequential_21/lstm_22/lstm_cell_22/MatMul_2:product:03sequential_21/lstm_22/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Н
,sequential_21/lstm_22/lstm_cell_22/BiasAdd_3BiasAdd5sequential_21/lstm_22/lstm_cell_22/MatMul_3:product:03sequential_21/lstm_22/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         └
(sequential_21/lstm_22/lstm_cell_22/mul_4Mul$sequential_21/lstm_22/zeros:output:07sequential_21/lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         └
(sequential_21/lstm_22/lstm_cell_22/mul_5Mul$sequential_21/lstm_22/zeros:output:07sequential_21/lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         └
(sequential_21/lstm_22/lstm_cell_22/mul_6Mul$sequential_21/lstm_22/zeros:output:07sequential_21/lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         └
(sequential_21/lstm_22/lstm_cell_22/mul_7Mul$sequential_21/lstm_22/zeros:output:07sequential_21/lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         г
1sequential_21/lstm_22/lstm_cell_22/ReadVariableOpReadVariableOp:sequential_21_lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0Є
6sequential_21/lstm_22/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Ѕ
8sequential_21/lstm_22/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ѕ
8sequential_21/lstm_22/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      џ
0sequential_21/lstm_22/lstm_cell_22/strided_sliceStridedSlice9sequential_21/lstm_22/lstm_cell_22/ReadVariableOp:value:0?sequential_21/lstm_22/lstm_cell_22/strided_slice/stack:output:0Asequential_21/lstm_22/lstm_cell_22/strided_slice/stack_1:output:0Asequential_21/lstm_22/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskл
+sequential_21/lstm_22/lstm_cell_22/MatMul_4MatMul,sequential_21/lstm_22/lstm_cell_22/mul_4:z:09sequential_21/lstm_22/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         ═
&sequential_21/lstm_22/lstm_cell_22/addAddV23sequential_21/lstm_22/lstm_cell_22/BiasAdd:output:05sequential_21/lstm_22/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         Њ
*sequential_21/lstm_22/lstm_cell_22/SigmoidSigmoid*sequential_21/lstm_22/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         «
3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_1ReadVariableOp:sequential_21_lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0Ѕ
8sequential_21/lstm_22/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       І
:sequential_21/lstm_22/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       І
:sequential_21/lstm_22/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
2sequential_21/lstm_22/lstm_cell_22/strided_slice_1StridedSlice;sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_1:value:0Asequential_21/lstm_22/lstm_cell_22/strided_slice_1/stack:output:0Csequential_21/lstm_22/lstm_cell_22/strided_slice_1/stack_1:output:0Csequential_21/lstm_22/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskм
+sequential_21/lstm_22/lstm_cell_22/MatMul_5MatMul,sequential_21/lstm_22/lstm_cell_22/mul_5:z:0;sequential_21/lstm_22/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Л
(sequential_21/lstm_22/lstm_cell_22/add_1AddV25sequential_21/lstm_22/lstm_cell_22/BiasAdd_1:output:05sequential_21/lstm_22/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         Ќ
,sequential_21/lstm_22/lstm_cell_22/Sigmoid_1Sigmoid,sequential_21/lstm_22/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ╗
(sequential_21/lstm_22/lstm_cell_22/mul_8Mul0sequential_21/lstm_22/lstm_cell_22/Sigmoid_1:y:0&sequential_21/lstm_22/zeros_1:output:0*
T0*'
_output_shapes
:         «
3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_2ReadVariableOp:sequential_21_lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0Ѕ
8sequential_21/lstm_22/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       І
:sequential_21/lstm_22/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       І
:sequential_21/lstm_22/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
2sequential_21/lstm_22/lstm_cell_22/strided_slice_2StridedSlice;sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_2:value:0Asequential_21/lstm_22/lstm_cell_22/strided_slice_2/stack:output:0Csequential_21/lstm_22/lstm_cell_22/strided_slice_2/stack_1:output:0Csequential_21/lstm_22/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskм
+sequential_21/lstm_22/lstm_cell_22/MatMul_6MatMul,sequential_21/lstm_22/lstm_cell_22/mul_6:z:0;sequential_21/lstm_22/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Л
(sequential_21/lstm_22/lstm_cell_22/add_2AddV25sequential_21/lstm_22/lstm_cell_22/BiasAdd_2:output:05sequential_21/lstm_22/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         Ќ
,sequential_21/lstm_22/lstm_cell_22/Sigmoid_2Sigmoid,sequential_21/lstm_22/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         ├
(sequential_21/lstm_22/lstm_cell_22/mul_9Mul.sequential_21/lstm_22/lstm_cell_22/Sigmoid:y:00sequential_21/lstm_22/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         ┐
(sequential_21/lstm_22/lstm_cell_22/add_3AddV2,sequential_21/lstm_22/lstm_cell_22/mul_8:z:0,sequential_21/lstm_22/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         «
3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_3ReadVariableOp:sequential_21_lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0Ѕ
8sequential_21/lstm_22/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       І
:sequential_21/lstm_22/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        І
:sequential_21/lstm_22/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ц
2sequential_21/lstm_22/lstm_cell_22/strided_slice_3StridedSlice;sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_3:value:0Asequential_21/lstm_22/lstm_cell_22/strided_slice_3/stack:output:0Csequential_21/lstm_22/lstm_cell_22/strided_slice_3/stack_1:output:0Csequential_21/lstm_22/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskм
+sequential_21/lstm_22/lstm_cell_22/MatMul_7MatMul,sequential_21/lstm_22/lstm_cell_22/mul_7:z:0;sequential_21/lstm_22/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Л
(sequential_21/lstm_22/lstm_cell_22/add_4AddV25sequential_21/lstm_22/lstm_cell_22/BiasAdd_3:output:05sequential_21/lstm_22/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         Ќ
,sequential_21/lstm_22/lstm_cell_22/Sigmoid_3Sigmoid,sequential_21/lstm_22/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         Ќ
,sequential_21/lstm_22/lstm_cell_22/Sigmoid_4Sigmoid,sequential_21/lstm_22/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         к
)sequential_21/lstm_22/lstm_cell_22/mul_10Mul0sequential_21/lstm_22/lstm_cell_22/Sigmoid_3:y:00sequential_21/lstm_22/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         ё
3sequential_21/lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Щ
%sequential_21/lstm_22/TensorArrayV2_1TensorListReserve<sequential_21/lstm_22/TensorArrayV2_1/element_shape:output:0.sequential_21/lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм\
sequential_21/lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_21/lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         j
(sequential_21/lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : г
sequential_21/lstm_22/whileWhile1sequential_21/lstm_22/while/loop_counter:output:07sequential_21/lstm_22/while/maximum_iterations:output:0#sequential_21/lstm_22/time:output:0.sequential_21/lstm_22/TensorArrayV2_1:handle:0$sequential_21/lstm_22/zeros:output:0&sequential_21/lstm_22/zeros_1:output:0.sequential_21/lstm_22/strided_slice_1:output:0Msequential_21/lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_21_lstm_22_lstm_cell_22_split_readvariableop_resourceBsequential_21_lstm_22_lstm_cell_22_split_1_readvariableop_resource:sequential_21_lstm_22_lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_21_lstm_22_while_body_440574*3
cond+R)
'sequential_21_lstm_22_while_cond_440573*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ќ
Fsequential_21/lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ё
8sequential_21/lstm_22/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_21/lstm_22/while:output:3Osequential_21/lstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0~
+sequential_21/lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         w
-sequential_21/lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_21/lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ш
%sequential_21/lstm_22/strided_slice_3StridedSliceAsequential_21/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:04sequential_21/lstm_22/strided_slice_3/stack:output:06sequential_21/lstm_22/strided_slice_3/stack_1:output:06sequential_21/lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_mask{
&sequential_21/lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          п
!sequential_21/lstm_22/transpose_1	TransposeAsequential_21/lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_21/lstm_22/transpose_1/perm:output:0*
T0*+
_output_shapes
:         q
sequential_21/lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    б
,sequential_21/dense_67/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_67_matmul_readvariableop_resource*
_output_shapes

:*
dtype0┐
sequential_21/dense_67/MatMulMatMul.sequential_21/lstm_22/strided_slice_3:output:04sequential_21/dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_21/dense_67/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_67_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_21/dense_67/BiasAddBiasAdd'sequential_21/dense_67/MatMul:product:05sequential_21/dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         б
,sequential_21/dense_68/MatMul/ReadVariableOpReadVariableOp5sequential_21_dense_68_matmul_readvariableop_resource*
_output_shapes

:*
dtype0И
sequential_21/dense_68/MatMulMatMul'sequential_21/dense_67/BiasAdd:output:04sequential_21/dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         а
-sequential_21/dense_68/BiasAdd/ReadVariableOpReadVariableOp6sequential_21_dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0╗
sequential_21/dense_68/BiasAddBiasAdd'sequential_21/dense_68/MatMul:product:05sequential_21/dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         v
IdentityIdentity'sequential_21/dense_68/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         л
NoOpNoOp.^sequential_21/dense_66/BiasAdd/ReadVariableOp0^sequential_21/dense_66/Tensordot/ReadVariableOp.^sequential_21/dense_67/BiasAdd/ReadVariableOp-^sequential_21/dense_67/MatMul/ReadVariableOp.^sequential_21/dense_68/BiasAdd/ReadVariableOp-^sequential_21/dense_68/MatMul/ReadVariableOp2^sequential_21/lstm_22/lstm_cell_22/ReadVariableOp4^sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_14^sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_24^sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_38^sequential_21/lstm_22/lstm_cell_22/split/ReadVariableOp:^sequential_21/lstm_22/lstm_cell_22/split_1/ReadVariableOp^sequential_21/lstm_22/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 2^
-sequential_21/dense_66/BiasAdd/ReadVariableOp-sequential_21/dense_66/BiasAdd/ReadVariableOp2b
/sequential_21/dense_66/Tensordot/ReadVariableOp/sequential_21/dense_66/Tensordot/ReadVariableOp2^
-sequential_21/dense_67/BiasAdd/ReadVariableOp-sequential_21/dense_67/BiasAdd/ReadVariableOp2\
,sequential_21/dense_67/MatMul/ReadVariableOp,sequential_21/dense_67/MatMul/ReadVariableOp2^
-sequential_21/dense_68/BiasAdd/ReadVariableOp-sequential_21/dense_68/BiasAdd/ReadVariableOp2\
,sequential_21/dense_68/MatMul/ReadVariableOp,sequential_21/dense_68/MatMul/ReadVariableOp2f
1sequential_21/lstm_22/lstm_cell_22/ReadVariableOp1sequential_21/lstm_22/lstm_cell_22/ReadVariableOp2j
3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_13sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_12j
3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_23sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_22j
3sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_33sequential_21/lstm_22/lstm_cell_22/ReadVariableOp_32r
7sequential_21/lstm_22/lstm_cell_22/split/ReadVariableOp7sequential_21/lstm_22/lstm_cell_22/split/ReadVariableOp2v
9sequential_21/lstm_22/lstm_cell_22/split_1/ReadVariableOp9sequential_21/lstm_22/lstm_cell_22/split_1/ReadVariableOp2:
sequential_21/lstm_22/whilesequential_21/lstm_22/while:[ W
+
_output_shapes
:         
(
_user_specified_namedense_66_input
Ч	
┌
.__inference_sequential_21_layer_call_fn_442203

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identityѕбStatefulPartitionedCall╣
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8ѓ *R
fMRK
I__inference_sequential_21_layer_call_and_return_conditional_losses_442055o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
х
├
while_cond_443723
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_443723___redundant_placeholder04
0while_while_cond_443723___redundant_placeholder14
0while_while_cond_443723___redundant_placeholder24
0while_while_cond_443723___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
ћѓ
Т
C__inference_lstm_22_layer_call_and_return_conditional_losses_443244
inputs_0<
*lstm_cell_22_split_readvariableop_resource::
,lstm_cell_22_split_1_readvariableop_resource:6
$lstm_cell_22_readvariableop_resource:
identityѕбlstm_cell_22/ReadVariableOpбlstm_cell_22/ReadVariableOp_1бlstm_cell_22/ReadVariableOp_2бlstm_cell_22/ReadVariableOp_3б!lstm_cell_22/split/ReadVariableOpб#lstm_cell_22/split_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskd
lstm_cell_22/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ъ
lstm_cell_22/ones_likeFill%lstm_cell_22/ones_like/Shape:output:0%lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         \
lstm_cell_22/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ц
lstm_cell_22/ones_like_1Fill'lstm_cell_22/ones_like_1/Shape:output:0'lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mulMulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_1Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_2Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/mul_3Mulstrided_slice_2:output:0lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
!lstm_cell_22/split/ReadVariableOpReadVariableOp*lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0┼
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0)lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitѓ
lstm_cell_22/MatMulMatMullstm_cell_22/mul:z:0lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_1MatMullstm_cell_22/mul_1:z:0lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_2MatMullstm_cell_22/mul_2:z:0lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_3MatMullstm_cell_22/mul_3:z:0lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         `
lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ї
#lstm_cell_22/split_1/ReadVariableOpReadVariableOp,lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0╗
lstm_cell_22/split_1Split'lstm_cell_22/split_1/split_dim:output:0+lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЈ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/MatMul:product:0lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_1BiasAddlstm_cell_22/MatMul_1:product:0lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_2BiasAddlstm_cell_22/MatMul_2:product:0lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_3BiasAddlstm_cell_22/MatMul_3:product:0lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_4Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_5Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_6Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ~
lstm_cell_22/mul_7Mulzeros:output:0!lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ђ
lstm_cell_22/ReadVariableOpReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      г
lstm_cell_22/strided_sliceStridedSlice#lstm_cell_22/ReadVariableOp:value:0)lstm_cell_22/strided_slice/stack:output:0+lstm_cell_22/strided_slice/stack_1:output:0+lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskј
lstm_cell_22/MatMul_4MatMullstm_cell_22/mul_4:z:0#lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         І
lstm_cell_22/addAddV2lstm_cell_22/BiasAdd:output:0lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         g
lstm_cell_22/SigmoidSigmoidlstm_cell_22/add:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_1ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_1StridedSlice%lstm_cell_22/ReadVariableOp_1:value:0+lstm_cell_22/strided_slice_1/stack:output:0-lstm_cell_22/strided_slice_1/stack_1:output:0-lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_5MatMullstm_cell_22/mul_5:z:0%lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_1AddV2lstm_cell_22/BiasAdd_1:output:0lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         y
lstm_cell_22/mul_8Mullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_2ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_2StridedSlice%lstm_cell_22/ReadVariableOp_2:value:0+lstm_cell_22/strided_slice_2/stack:output:0-lstm_cell_22/strided_slice_2/stack_1:output:0-lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_6MatMullstm_cell_22/mul_6:z:0%lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_2AddV2lstm_cell_22/BiasAdd_2:output:0lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell_22/mul_9Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         }
lstm_cell_22/add_3AddV2lstm_cell_22/mul_8:z:0lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_3ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_3StridedSlice%lstm_cell_22/ReadVariableOp_3:value:0+lstm_cell_22/strided_slice_3/stack:output:0-lstm_cell_22/strided_slice_3/stack_1:output:0-lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_7MatMullstm_cell_22/mul_7:z:0%lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_4AddV2lstm_cell_22/BiasAdd_3:output:0lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_3Sigmoidlstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_4Sigmoidlstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mul_10Mullstm_cell_22/Sigmoid_3:y:0lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_22_split_readvariableop_resource,lstm_cell_22_split_1_readvariableop_resource$lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_443110*
condR
while_cond_443109*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp^lstm_cell_22/ReadVariableOp^lstm_cell_22/ReadVariableOp_1^lstm_cell_22/ReadVariableOp_2^lstm_cell_22/ReadVariableOp_3"^lstm_cell_22/split/ReadVariableOp$^lstm_cell_22/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2:
lstm_cell_22/ReadVariableOplstm_cell_22/ReadVariableOp2>
lstm_cell_22/ReadVariableOp_1lstm_cell_22/ReadVariableOp_12>
lstm_cell_22/ReadVariableOp_2lstm_cell_22/ReadVariableOp_22>
lstm_cell_22/ReadVariableOp_3lstm_cell_22/ReadVariableOp_32F
!lstm_cell_22/split/ReadVariableOp!lstm_cell_22/split/ReadVariableOp2J
#lstm_cell_22/split_1/ReadVariableOp#lstm_cell_22/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
н─
ъ	
while_body_444031
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_22_split_readvariableop_resource_0:B
4while_lstm_cell_22_split_1_readvariableop_resource_0:>
,while_lstm_cell_22_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_22_split_readvariableop_resource:@
2while_lstm_cell_22_split_1_readvariableop_resource:<
*while_lstm_cell_22_readvariableop_resource:ѕб!while/lstm_cell_22/ReadVariableOpб#while/lstm_cell_22/ReadVariableOp_1б#while/lstm_cell_22/ReadVariableOp_2б#while/lstm_cell_22/ReadVariableOp_3б'while/lstm_cell_22/split/ReadVariableOpб)while/lstm_cell_22/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ѓ
"while/lstm_cell_22/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?░
while/lstm_cell_22/ones_likeFill+while/lstm_cell_22/ones_like/Shape:output:0+while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         e
 while/lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Е
while/lstm_cell_22/dropout/MulMul%while/lstm_cell_22/ones_like:output:0)while/lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         u
 while/lstm_cell_22/dropout/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:▓
7while/lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0n
)while/lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=▀
'while/lstm_cell_22/dropout/GreaterEqualGreaterEqual@while/lstm_cell_22/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ћ
while/lstm_cell_22/dropout/CastCast+while/lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         б
 while/lstm_cell_22/dropout/Mul_1Mul"while/lstm_cell_22/dropout/Mul:z:0#while/lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_1/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_1/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_1/CastCast-while/lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_1/Mul_1Mul$while/lstm_cell_22/dropout_1/Mul:z:0%while/lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_2/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_2/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_2/CastCast-while/lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_2/Mul_1Mul$while/lstm_cell_22/dropout_2/Mul:z:0%while/lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_3/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_3/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_3/CastCast-while/lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_3/Mul_1Mul$while/lstm_cell_22/dropout_3/Mul:z:0%while/lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         g
$while/lstm_cell_22/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
while/lstm_cell_22/ones_like_1Fill-while/lstm_cell_22/ones_like_1/Shape:output:0-while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_4/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_4/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_4/CastCast-while/lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_4/Mul_1Mul$while/lstm_cell_22/dropout_4/Mul:z:0%while/lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_5/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_5/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_5/CastCast-while/lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_5/Mul_1Mul$while/lstm_cell_22/dropout_5/Mul:z:0%while/lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_6/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_6/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_6/CastCast-while/lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_6/Mul_1Mul$while/lstm_cell_22/dropout_6/Mul:z:0%while/lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_7/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_7/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_7/CastCast-while/lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_7/Mul_1Mul$while/lstm_cell_22/dropout_7/Mul:z:0%while/lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         Д
while/lstm_cell_22/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :џ
'while/lstm_cell_22/split/ReadVariableOpReadVariableOp2while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0О
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitћ
while/lstm_cell_22/MatMulMatMulwhile/lstm_cell_22/mul:z:0!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_1MatMulwhile/lstm_cell_22/mul_1:z:0!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_2MatMulwhile/lstm_cell_22/mul_2:z:0!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_3MatMulwhile/lstm_cell_22/mul_3:z:0!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         f
$while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : џ
)while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0═
while/lstm_cell_22/split_1Split-while/lstm_cell_22/split_1/split_dim:output:01while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitА
while/lstm_cell_22/BiasAddBiasAdd#while/lstm_cell_22/MatMul:product:0#while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_1BiasAdd%while/lstm_cell_22/MatMul_1:product:0#while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_2BiasAdd%while/lstm_cell_22/MatMul_2:product:0#while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_3BiasAdd%while/lstm_cell_22/MatMul_3:product:0#while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_4Mulwhile_placeholder_2&while/lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_5Mulwhile_placeholder_2&while/lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_6Mulwhile_placeholder_2&while/lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_7Mulwhile_placeholder_2&while/lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         ј
!while/lstm_cell_22/ReadVariableOpReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_22/strided_sliceStridedSlice)while/lstm_cell_22/ReadVariableOp:value:0/while/lstm_cell_22/strided_slice/stack:output:01while/lstm_cell_22/strided_slice/stack_1:output:01while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskа
while/lstm_cell_22/MatMul_4MatMulwhile/lstm_cell_22/mul_4:z:0)while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/BiasAdd:output:0%while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         s
while/lstm_cell_22/SigmoidSigmoidwhile/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_1ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_1StridedSlice+while/lstm_cell_22/ReadVariableOp_1:value:01while/lstm_cell_22/strided_slice_1/stack:output:03while/lstm_cell_22/strided_slice_1/stack_1:output:03while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_5MatMulwhile/lstm_cell_22/mul_5:z:0+while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_1AddV2%while/lstm_cell_22/BiasAdd_1:output:0%while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_1Sigmoidwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ѕ
while/lstm_cell_22/mul_8Mul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_2ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_2StridedSlice+while/lstm_cell_22/ReadVariableOp_2:value:01while/lstm_cell_22/strided_slice_2/stack:output:03while/lstm_cell_22/strided_slice_2/stack_1:output:03while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_6MatMulwhile/lstm_cell_22/mul_6:z:0+while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_2AddV2%while/lstm_cell_22/BiasAdd_2:output:0%while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_2Sigmoidwhile/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell_22/mul_9Mulwhile/lstm_cell_22/Sigmoid:y:0 while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/add_3AddV2while/lstm_cell_22/mul_8:z:0while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_3ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_3StridedSlice+while/lstm_cell_22/ReadVariableOp_3:value:01while/lstm_cell_22/strided_slice_3/stack:output:03while/lstm_cell_22/strided_slice_3/stack_1:output:03while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_7MatMulwhile/lstm_cell_22/mul_7:z:0+while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_4AddV2%while/lstm_cell_22/BiasAdd_3:output:0%while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_3Sigmoidwhile/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_4Sigmoidwhile/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ќ
while/lstm_cell_22/mul_10Mul while/lstm_cell_22/Sigmoid_3:y:0 while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         к
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмz
while/Identity_4Identitywhile/lstm_cell_22/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         И

while/NoOpNoOp"^while/lstm_cell_22/ReadVariableOp$^while/lstm_cell_22/ReadVariableOp_1$^while/lstm_cell_22/ReadVariableOp_2$^while/lstm_cell_22/ReadVariableOp_3(^while/lstm_cell_22/split/ReadVariableOp*^while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_22_readvariableop_resource,while_lstm_cell_22_readvariableop_resource_0"j
2while_lstm_cell_22_split_1_readvariableop_resource4while_lstm_cell_22_split_1_readvariableop_resource_0"f
0while_lstm_cell_22_split_readvariableop_resource2while_lstm_cell_22_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2F
!while/lstm_cell_22/ReadVariableOp!while/lstm_cell_22/ReadVariableOp2J
#while/lstm_cell_22/ReadVariableOp_1#while/lstm_cell_22/ReadVariableOp_12J
#while/lstm_cell_22/ReadVariableOp_2#while/lstm_cell_22/ReadVariableOp_22J
#while/lstm_cell_22/ReadVariableOp_3#while/lstm_cell_22/ReadVariableOp_32R
'while/lstm_cell_22/split/ReadVariableOp'while/lstm_cell_22/split/ReadVariableOp2V
)while/lstm_cell_22/split_1/ReadVariableOp)while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_441786
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_441786___redundant_placeholder04
0while_while_cond_441786___redundant_placeholder14
0while_while_cond_441786___redundant_placeholder24
0while_while_cond_441786___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
х
├
while_cond_443416
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_443416___redundant_placeholder04
0while_while_cond_443416___redundant_placeholder14
0while_while_cond_443416___redundant_placeholder24
0while_while_cond_443416___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
м
ќ
)__inference_dense_66_layer_call_fn_442927

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCallП
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_66_layer_call_and_return_conditional_losses_441271s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ГD
е
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_444383

inputs
states_0
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2ѕбReadVariableOpбReadVariableOp_1бReadVariableOp_2бReadVariableOp_3бsplit/ReadVariableOpбsplit_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:         I
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:         X
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:         Z
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:         Z
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:         Z
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:         Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0ъ
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:         _
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:         _
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:         _
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:         S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0ћ
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:         l
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:         l
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:         l
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:         ^
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:         ^
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:         ^
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:         ^
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:         f
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0d
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        f
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       f
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      в
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskg
MatMul_4MatMul	mul_4:z:0strided_slice:output:0*
T0*'
_output_shapes
:         d
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:         M
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_5MatMul	mul_5:z:0strided_slice_1:output:0*
T0*'
_output_shapes
:         h
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:         W
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         h
ReadVariableOp_2ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_2StridedSliceReadVariableOp_2:value:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_6MatMul	mul_6:z:0strided_slice_2:output:0*
T0*'
_output_shapes
:         h
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:         Z
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:         V
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:         h
ReadVariableOp_3ReadVariableOpreadvariableop_resource*
_output_shapes

:*
dtype0f
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       h
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        h
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ш
strided_slice_3StridedSliceReadVariableOp_3:value:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maski
MatMul_7MatMul	mul_7:z:0strided_slice_3:output:0*
T0*'
_output_shapes
:         h
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:         Q
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:         Q
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:         ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:         Y
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         [

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:         Z

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:         └
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:         :         :         : : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:QM
'
_output_shapes
:         
"
_user_specified_name
states/0:QM
'
_output_shapes
:         
"
_user_specified_name
states/1
▄Ц
б
'sequential_21_lstm_22_while_body_440574H
Dsequential_21_lstm_22_while_sequential_21_lstm_22_while_loop_counterN
Jsequential_21_lstm_22_while_sequential_21_lstm_22_while_maximum_iterations+
'sequential_21_lstm_22_while_placeholder-
)sequential_21_lstm_22_while_placeholder_1-
)sequential_21_lstm_22_while_placeholder_2-
)sequential_21_lstm_22_while_placeholder_3G
Csequential_21_lstm_22_while_sequential_21_lstm_22_strided_slice_1_0Ѓ
sequential_21_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_21_lstm_22_tensorarrayunstack_tensorlistfromtensor_0Z
Hsequential_21_lstm_22_while_lstm_cell_22_split_readvariableop_resource_0:X
Jsequential_21_lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0:T
Bsequential_21_lstm_22_while_lstm_cell_22_readvariableop_resource_0:(
$sequential_21_lstm_22_while_identity*
&sequential_21_lstm_22_while_identity_1*
&sequential_21_lstm_22_while_identity_2*
&sequential_21_lstm_22_while_identity_3*
&sequential_21_lstm_22_while_identity_4*
&sequential_21_lstm_22_while_identity_5E
Asequential_21_lstm_22_while_sequential_21_lstm_22_strided_slice_1Ђ
}sequential_21_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_21_lstm_22_tensorarrayunstack_tensorlistfromtensorX
Fsequential_21_lstm_22_while_lstm_cell_22_split_readvariableop_resource:V
Hsequential_21_lstm_22_while_lstm_cell_22_split_1_readvariableop_resource:R
@sequential_21_lstm_22_while_lstm_cell_22_readvariableop_resource:ѕб7sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOpб9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_1б9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_2б9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_3б=sequential_21/lstm_22/while/lstm_cell_22/split/ReadVariableOpб?sequential_21/lstm_22/while/lstm_cell_22/split_1/ReadVariableOpъ
Msequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ћ
?sequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_21_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_21_lstm_22_tensorarrayunstack_tensorlistfromtensor_0'sequential_21_lstm_22_while_placeholderVsequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0«
8sequential_21/lstm_22/while/lstm_cell_22/ones_like/ShapeShapeFsequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:}
8sequential_21/lstm_22/while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Ы
2sequential_21/lstm_22/while/lstm_cell_22/ones_likeFillAsequential_21/lstm_22/while/lstm_cell_22/ones_like/Shape:output:0Asequential_21/lstm_22/while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         Њ
:sequential_21/lstm_22/while/lstm_cell_22/ones_like_1/ShapeShape)sequential_21_lstm_22_while_placeholder_2*
T0*
_output_shapes
:
:sequential_21/lstm_22/while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Э
4sequential_21/lstm_22/while/lstm_cell_22/ones_like_1FillCsequential_21/lstm_22/while/lstm_cell_22/ones_like_1/Shape:output:0Csequential_21/lstm_22/while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         Ж
,sequential_21/lstm_22/while/lstm_cell_22/mulMulFsequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_21/lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         В
.sequential_21/lstm_22/while/lstm_cell_22/mul_1MulFsequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_21/lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         В
.sequential_21/lstm_22/while/lstm_cell_22/mul_2MulFsequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_21/lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         В
.sequential_21/lstm_22/while/lstm_cell_22/mul_3MulFsequential_21/lstm_22/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_21/lstm_22/while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         z
8sequential_21/lstm_22/while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :к
=sequential_21/lstm_22/while/lstm_cell_22/split/ReadVariableOpReadVariableOpHsequential_21_lstm_22_while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0Ў
.sequential_21/lstm_22/while/lstm_cell_22/splitSplitAsequential_21/lstm_22/while/lstm_cell_22/split/split_dim:output:0Esequential_21/lstm_22/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitо
/sequential_21/lstm_22/while/lstm_cell_22/MatMulMatMul0sequential_21/lstm_22/while/lstm_cell_22/mul:z:07sequential_21/lstm_22/while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ┌
1sequential_21/lstm_22/while/lstm_cell_22/MatMul_1MatMul2sequential_21/lstm_22/while/lstm_cell_22/mul_1:z:07sequential_21/lstm_22/while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ┌
1sequential_21/lstm_22/while/lstm_cell_22/MatMul_2MatMul2sequential_21/lstm_22/while/lstm_cell_22/mul_2:z:07sequential_21/lstm_22/while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ┌
1sequential_21/lstm_22/while/lstm_cell_22/MatMul_3MatMul2sequential_21/lstm_22/while/lstm_cell_22/mul_3:z:07sequential_21/lstm_22/while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         |
:sequential_21/lstm_22/while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : к
?sequential_21/lstm_22/while/lstm_cell_22/split_1/ReadVariableOpReadVariableOpJsequential_21_lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0Ј
0sequential_21/lstm_22/while/lstm_cell_22/split_1SplitCsequential_21/lstm_22/while/lstm_cell_22/split_1/split_dim:output:0Gsequential_21/lstm_22/while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitс
0sequential_21/lstm_22/while/lstm_cell_22/BiasAddBiasAdd9sequential_21/lstm_22/while/lstm_cell_22/MatMul:product:09sequential_21/lstm_22/while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         у
2sequential_21/lstm_22/while/lstm_cell_22/BiasAdd_1BiasAdd;sequential_21/lstm_22/while/lstm_cell_22/MatMul_1:product:09sequential_21/lstm_22/while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         у
2sequential_21/lstm_22/while/lstm_cell_22/BiasAdd_2BiasAdd;sequential_21/lstm_22/while/lstm_cell_22/MatMul_2:product:09sequential_21/lstm_22/while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         у
2sequential_21/lstm_22/while/lstm_cell_22/BiasAdd_3BiasAdd;sequential_21/lstm_22/while/lstm_cell_22/MatMul_3:product:09sequential_21/lstm_22/while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         Л
.sequential_21/lstm_22/while/lstm_cell_22/mul_4Mul)sequential_21_lstm_22_while_placeholder_2=sequential_21/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Л
.sequential_21/lstm_22/while/lstm_cell_22/mul_5Mul)sequential_21_lstm_22_while_placeholder_2=sequential_21/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Л
.sequential_21/lstm_22/while/lstm_cell_22/mul_6Mul)sequential_21_lstm_22_while_placeholder_2=sequential_21/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Л
.sequential_21/lstm_22/while/lstm_cell_22/mul_7Mul)sequential_21_lstm_22_while_placeholder_2=sequential_21/lstm_22/while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ║
7sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOpReadVariableOpBsequential_21_lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ї
<sequential_21/lstm_22/while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        Ј
>sequential_21/lstm_22/while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Ј
>sequential_21/lstm_22/while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      И
6sequential_21/lstm_22/while/lstm_cell_22/strided_sliceStridedSlice?sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp:value:0Esequential_21/lstm_22/while/lstm_cell_22/strided_slice/stack:output:0Gsequential_21/lstm_22/while/lstm_cell_22/strided_slice/stack_1:output:0Gsequential_21/lstm_22/while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskР
1sequential_21/lstm_22/while/lstm_cell_22/MatMul_4MatMul2sequential_21/lstm_22/while/lstm_cell_22/mul_4:z:0?sequential_21/lstm_22/while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         ▀
,sequential_21/lstm_22/while/lstm_cell_22/addAddV29sequential_21/lstm_22/while/lstm_cell_22/BiasAdd:output:0;sequential_21/lstm_22/while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         Ъ
0sequential_21/lstm_22/while/lstm_cell_22/SigmoidSigmoid0sequential_21/lstm_22/while/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         ╝
9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_1ReadVariableOpBsequential_21_lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ј
>sequential_21/lstm_22/while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       Љ
@sequential_21/lstm_22/while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Љ
@sequential_21/lstm_22/while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_21/lstm_22/while/lstm_cell_22/strided_slice_1StridedSliceAsequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_1:value:0Gsequential_21/lstm_22/while/lstm_cell_22/strided_slice_1/stack:output:0Isequential_21/lstm_22/while/lstm_cell_22/strided_slice_1/stack_1:output:0Isequential_21/lstm_22/while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskС
1sequential_21/lstm_22/while/lstm_cell_22/MatMul_5MatMul2sequential_21/lstm_22/while/lstm_cell_22/mul_5:z:0Asequential_21/lstm_22/while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         с
.sequential_21/lstm_22/while/lstm_cell_22/add_1AddV2;sequential_21/lstm_22/while/lstm_cell_22/BiasAdd_1:output:0;sequential_21/lstm_22/while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         Б
2sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_1Sigmoid2sequential_21/lstm_22/while/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ╩
.sequential_21/lstm_22/while/lstm_cell_22/mul_8Mul6sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_1:y:0)sequential_21_lstm_22_while_placeholder_3*
T0*'
_output_shapes
:         ╝
9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_2ReadVariableOpBsequential_21_lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ј
>sequential_21/lstm_22/while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       Љ
@sequential_21/lstm_22/while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       Љ
@sequential_21/lstm_22/while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_21/lstm_22/while/lstm_cell_22/strided_slice_2StridedSliceAsequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_2:value:0Gsequential_21/lstm_22/while/lstm_cell_22/strided_slice_2/stack:output:0Isequential_21/lstm_22/while/lstm_cell_22/strided_slice_2/stack_1:output:0Isequential_21/lstm_22/while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskС
1sequential_21/lstm_22/while/lstm_cell_22/MatMul_6MatMul2sequential_21/lstm_22/while/lstm_cell_22/mul_6:z:0Asequential_21/lstm_22/while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         с
.sequential_21/lstm_22/while/lstm_cell_22/add_2AddV2;sequential_21/lstm_22/while/lstm_cell_22/BiasAdd_2:output:0;sequential_21/lstm_22/while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         Б
2sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_2Sigmoid2sequential_21/lstm_22/while/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Н
.sequential_21/lstm_22/while/lstm_cell_22/mul_9Mul4sequential_21/lstm_22/while/lstm_cell_22/Sigmoid:y:06sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Л
.sequential_21/lstm_22/while/lstm_cell_22/add_3AddV22sequential_21/lstm_22/while/lstm_cell_22/mul_8:z:02sequential_21/lstm_22/while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         ╝
9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_3ReadVariableOpBsequential_21_lstm_22_while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0Ј
>sequential_21/lstm_22/while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       Љ
@sequential_21/lstm_22/while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        Љ
@sequential_21/lstm_22/while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ┬
8sequential_21/lstm_22/while/lstm_cell_22/strided_slice_3StridedSliceAsequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_3:value:0Gsequential_21/lstm_22/while/lstm_cell_22/strided_slice_3/stack:output:0Isequential_21/lstm_22/while/lstm_cell_22/strided_slice_3/stack_1:output:0Isequential_21/lstm_22/while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskС
1sequential_21/lstm_22/while/lstm_cell_22/MatMul_7MatMul2sequential_21/lstm_22/while/lstm_cell_22/mul_7:z:0Asequential_21/lstm_22/while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         с
.sequential_21/lstm_22/while/lstm_cell_22/add_4AddV2;sequential_21/lstm_22/while/lstm_cell_22/BiasAdd_3:output:0;sequential_21/lstm_22/while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         Б
2sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_3Sigmoid2sequential_21/lstm_22/while/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         Б
2sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_4Sigmoid2sequential_21/lstm_22/while/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         п
/sequential_21/lstm_22/while/lstm_cell_22/mul_10Mul6sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_3:y:06sequential_21/lstm_22/while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         ъ
@sequential_21/lstm_22/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_21_lstm_22_while_placeholder_1'sequential_21_lstm_22_while_placeholder3sequential_21/lstm_22/while/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмc
!sequential_21/lstm_22/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :ъ
sequential_21/lstm_22/while/addAddV2'sequential_21_lstm_22_while_placeholder*sequential_21/lstm_22/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_21/lstm_22/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :┐
!sequential_21/lstm_22/while/add_1AddV2Dsequential_21_lstm_22_while_sequential_21_lstm_22_while_loop_counter,sequential_21/lstm_22/while/add_1/y:output:0*
T0*
_output_shapes
: Џ
$sequential_21/lstm_22/while/IdentityIdentity%sequential_21/lstm_22/while/add_1:z:0!^sequential_21/lstm_22/while/NoOp*
T0*
_output_shapes
: ┬
&sequential_21/lstm_22/while/Identity_1IdentityJsequential_21_lstm_22_while_sequential_21_lstm_22_while_maximum_iterations!^sequential_21/lstm_22/while/NoOp*
T0*
_output_shapes
: Џ
&sequential_21/lstm_22/while/Identity_2Identity#sequential_21/lstm_22/while/add:z:0!^sequential_21/lstm_22/while/NoOp*
T0*
_output_shapes
: █
&sequential_21/lstm_22/while/Identity_3IdentityPsequential_21/lstm_22/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_21/lstm_22/while/NoOp*
T0*
_output_shapes
: :жУм╝
&sequential_21/lstm_22/while/Identity_4Identity3sequential_21/lstm_22/while/lstm_cell_22/mul_10:z:0!^sequential_21/lstm_22/while/NoOp*
T0*'
_output_shapes
:         ╗
&sequential_21/lstm_22/while/Identity_5Identity2sequential_21/lstm_22/while/lstm_cell_22/add_3:z:0!^sequential_21/lstm_22/while/NoOp*
T0*'
_output_shapes
:         м
 sequential_21/lstm_22/while/NoOpNoOp8^sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp:^sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_1:^sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_2:^sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_3>^sequential_21/lstm_22/while/lstm_cell_22/split/ReadVariableOp@^sequential_21/lstm_22/while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_21_lstm_22_while_identity-sequential_21/lstm_22/while/Identity:output:0"Y
&sequential_21_lstm_22_while_identity_1/sequential_21/lstm_22/while/Identity_1:output:0"Y
&sequential_21_lstm_22_while_identity_2/sequential_21/lstm_22/while/Identity_2:output:0"Y
&sequential_21_lstm_22_while_identity_3/sequential_21/lstm_22/while/Identity_3:output:0"Y
&sequential_21_lstm_22_while_identity_4/sequential_21/lstm_22/while/Identity_4:output:0"Y
&sequential_21_lstm_22_while_identity_5/sequential_21/lstm_22/while/Identity_5:output:0"є
@sequential_21_lstm_22_while_lstm_cell_22_readvariableop_resourceBsequential_21_lstm_22_while_lstm_cell_22_readvariableop_resource_0"ќ
Hsequential_21_lstm_22_while_lstm_cell_22_split_1_readvariableop_resourceJsequential_21_lstm_22_while_lstm_cell_22_split_1_readvariableop_resource_0"њ
Fsequential_21_lstm_22_while_lstm_cell_22_split_readvariableop_resourceHsequential_21_lstm_22_while_lstm_cell_22_split_readvariableop_resource_0"ѕ
Asequential_21_lstm_22_while_sequential_21_lstm_22_strided_slice_1Csequential_21_lstm_22_while_sequential_21_lstm_22_strided_slice_1_0"ђ
}sequential_21_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_21_lstm_22_tensorarrayunstack_tensorlistfromtensorsequential_21_lstm_22_while_tensorarrayv2read_tensorlistgetitem_sequential_21_lstm_22_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2r
7sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp7sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp2v
9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_19sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_12v
9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_29sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_22v
9sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_39sequential_21/lstm_22/while/lstm_cell_22/ReadVariableOp_32~
=sequential_21/lstm_22/while/lstm_cell_22/split/ReadVariableOp=sequential_21/lstm_22/while/lstm_cell_22/split/ReadVariableOp2ѓ
?sequential_21/lstm_22/while/lstm_cell_22/split_1/ReadVariableOp?sequential_21/lstm_22/while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ёv
ъ	
while_body_441385
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_22_split_readvariableop_resource_0:B
4while_lstm_cell_22_split_1_readvariableop_resource_0:>
,while_lstm_cell_22_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_22_split_readvariableop_resource:@
2while_lstm_cell_22_split_1_readvariableop_resource:<
*while_lstm_cell_22_readvariableop_resource:ѕб!while/lstm_cell_22/ReadVariableOpб#while/lstm_cell_22/ReadVariableOp_1б#while/lstm_cell_22/ReadVariableOp_2б#while/lstm_cell_22/ReadVariableOp_3б'while/lstm_cell_22/split/ReadVariableOpб)while/lstm_cell_22/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ѓ
"while/lstm_cell_22/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?░
while/lstm_cell_22/ones_likeFill+while/lstm_cell_22/ones_like/Shape:output:0+while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         g
$while/lstm_cell_22/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
while/lstm_cell_22/ones_like_1Fill-while/lstm_cell_22/ones_like_1/Shape:output:0-while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         е
while/lstm_cell_22/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :џ
'while/lstm_cell_22/split/ReadVariableOpReadVariableOp2while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0О
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitћ
while/lstm_cell_22/MatMulMatMulwhile/lstm_cell_22/mul:z:0!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_1MatMulwhile/lstm_cell_22/mul_1:z:0!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_2MatMulwhile/lstm_cell_22/mul_2:z:0!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_3MatMulwhile/lstm_cell_22/mul_3:z:0!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         f
$while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : џ
)while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0═
while/lstm_cell_22/split_1Split-while/lstm_cell_22/split_1/split_dim:output:01while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitА
while/lstm_cell_22/BiasAddBiasAdd#while/lstm_cell_22/MatMul:product:0#while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_1BiasAdd%while/lstm_cell_22/MatMul_1:product:0#while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_2BiasAdd%while/lstm_cell_22/MatMul_2:product:0#while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_3BiasAdd%while/lstm_cell_22/MatMul_3:product:0#while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_4Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_5Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_6Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_7Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ј
!while/lstm_cell_22/ReadVariableOpReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_22/strided_sliceStridedSlice)while/lstm_cell_22/ReadVariableOp:value:0/while/lstm_cell_22/strided_slice/stack:output:01while/lstm_cell_22/strided_slice/stack_1:output:01while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskа
while/lstm_cell_22/MatMul_4MatMulwhile/lstm_cell_22/mul_4:z:0)while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/BiasAdd:output:0%while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         s
while/lstm_cell_22/SigmoidSigmoidwhile/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_1ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_1StridedSlice+while/lstm_cell_22/ReadVariableOp_1:value:01while/lstm_cell_22/strided_slice_1/stack:output:03while/lstm_cell_22/strided_slice_1/stack_1:output:03while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_5MatMulwhile/lstm_cell_22/mul_5:z:0+while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_1AddV2%while/lstm_cell_22/BiasAdd_1:output:0%while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_1Sigmoidwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ѕ
while/lstm_cell_22/mul_8Mul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_2ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_2StridedSlice+while/lstm_cell_22/ReadVariableOp_2:value:01while/lstm_cell_22/strided_slice_2/stack:output:03while/lstm_cell_22/strided_slice_2/stack_1:output:03while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_6MatMulwhile/lstm_cell_22/mul_6:z:0+while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_2AddV2%while/lstm_cell_22/BiasAdd_2:output:0%while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_2Sigmoidwhile/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell_22/mul_9Mulwhile/lstm_cell_22/Sigmoid:y:0 while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/add_3AddV2while/lstm_cell_22/mul_8:z:0while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_3ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_3StridedSlice+while/lstm_cell_22/ReadVariableOp_3:value:01while/lstm_cell_22/strided_slice_3/stack:output:03while/lstm_cell_22/strided_slice_3/stack_1:output:03while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_7MatMulwhile/lstm_cell_22/mul_7:z:0+while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_4AddV2%while/lstm_cell_22/BiasAdd_3:output:0%while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_3Sigmoidwhile/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_4Sigmoidwhile/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ќ
while/lstm_cell_22/mul_10Mul while/lstm_cell_22/Sigmoid_3:y:0 while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         к
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмz
while/Identity_4Identitywhile/lstm_cell_22/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         И

while/NoOpNoOp"^while/lstm_cell_22/ReadVariableOp$^while/lstm_cell_22/ReadVariableOp_1$^while/lstm_cell_22/ReadVariableOp_2$^while/lstm_cell_22/ReadVariableOp_3(^while/lstm_cell_22/split/ReadVariableOp*^while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_22_readvariableop_resource,while_lstm_cell_22_readvariableop_resource_0"j
2while_lstm_cell_22_split_1_readvariableop_resource4while_lstm_cell_22_split_1_readvariableop_resource_0"f
0while_lstm_cell_22_split_readvariableop_resource2while_lstm_cell_22_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2F
!while/lstm_cell_22/ReadVariableOp!while/lstm_cell_22/ReadVariableOp2J
#while/lstm_cell_22/ReadVariableOp_1#while/lstm_cell_22/ReadVariableOp_12J
#while/lstm_cell_22/ReadVariableOp_2#while/lstm_cell_22/ReadVariableOp_22J
#while/lstm_cell_22/ReadVariableOp_3#while/lstm_cell_22/ReadVariableOp_32R
'while/lstm_cell_22/split/ReadVariableOp'while/lstm_cell_22/split/ReadVariableOp2V
)while/lstm_cell_22/split_1/ReadVariableOp)while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
 ╔
С
C__inference_lstm_22_layer_call_and_return_conditional_losses_441985

inputs<
*lstm_cell_22_split_readvariableop_resource::
,lstm_cell_22_split_1_readvariableop_resource:6
$lstm_cell_22_readvariableop_resource:
identityѕбlstm_cell_22/ReadVariableOpбlstm_cell_22/ReadVariableOp_1бlstm_cell_22/ReadVariableOp_2бlstm_cell_22/ReadVariableOp_3б!lstm_cell_22/split/ReadVariableOpб#lstm_cell_22/split_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:         D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskd
lstm_cell_22/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ъ
lstm_cell_22/ones_likeFill%lstm_cell_22/ones_like/Shape:output:0%lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         _
lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ќ
lstm_cell_22/dropout/MulMullstm_cell_22/ones_like:output:0#lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         i
lstm_cell_22/dropout/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:д
1lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0h
#lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=═
!lstm_cell_22/dropout/GreaterEqualGreaterEqual:lstm_cell_22/dropout/random_uniform/RandomUniform:output:0,lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ѕ
lstm_cell_22/dropout/CastCast%lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         љ
lstm_cell_22/dropout/Mul_1Mullstm_cell_22/dropout/Mul:z:0lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_1/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_1/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_1/GreaterEqualGreaterEqual<lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_1/CastCast'lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_1/Mul_1Mullstm_cell_22/dropout_1/Mul:z:0lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_2/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_2/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_2/GreaterEqualGreaterEqual<lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_2/CastCast'lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_2/Mul_1Mullstm_cell_22/dropout_2/Mul:z:0lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_3/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_3/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_3/GreaterEqualGreaterEqual<lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_3/CastCast'lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_3/Mul_1Mullstm_cell_22/dropout_3/Mul:z:0lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         \
lstm_cell_22/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ц
lstm_cell_22/ones_like_1Fill'lstm_cell_22/ones_like_1/Shape:output:0'lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_4/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_4/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_4/GreaterEqualGreaterEqual<lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_4/CastCast'lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_4/Mul_1Mullstm_cell_22/dropout_4/Mul:z:0lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_5/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_5/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_5/GreaterEqualGreaterEqual<lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_5/CastCast'lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_5/Mul_1Mullstm_cell_22/dropout_5/Mul:z:0lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_6/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_6/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_6/GreaterEqualGreaterEqual<lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_6/CastCast'lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_6/Mul_1Mullstm_cell_22/dropout_6/Mul:z:0lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_7/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_7/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_7/GreaterEqualGreaterEqual<lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_7/CastCast'lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_7/Mul_1Mullstm_cell_22/dropout_7/Mul:z:0lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         Ѓ
lstm_cell_22/mulMulstrided_slice_2:output:0lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_1Mulstrided_slice_2:output:0 lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_2Mulstrided_slice_2:output:0 lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_3Mulstrided_slice_2:output:0 lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         ^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
!lstm_cell_22/split/ReadVariableOpReadVariableOp*lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0┼
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0)lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitѓ
lstm_cell_22/MatMulMatMullstm_cell_22/mul:z:0lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_1MatMullstm_cell_22/mul_1:z:0lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_2MatMullstm_cell_22/mul_2:z:0lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_3MatMullstm_cell_22/mul_3:z:0lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         `
lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ї
#lstm_cell_22/split_1/ReadVariableOpReadVariableOp,lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0╗
lstm_cell_22/split_1Split'lstm_cell_22/split_1/split_dim:output:0+lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЈ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/MatMul:product:0lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_1BiasAddlstm_cell_22/MatMul_1:product:0lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_2BiasAddlstm_cell_22/MatMul_2:product:0lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_3BiasAddlstm_cell_22/MatMul_3:product:0lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_4Mulzeros:output:0 lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_5Mulzeros:output:0 lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_6Mulzeros:output:0 lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_7Mulzeros:output:0 lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_cell_22/ReadVariableOpReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      г
lstm_cell_22/strided_sliceStridedSlice#lstm_cell_22/ReadVariableOp:value:0)lstm_cell_22/strided_slice/stack:output:0+lstm_cell_22/strided_slice/stack_1:output:0+lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskј
lstm_cell_22/MatMul_4MatMullstm_cell_22/mul_4:z:0#lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         І
lstm_cell_22/addAddV2lstm_cell_22/BiasAdd:output:0lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         g
lstm_cell_22/SigmoidSigmoidlstm_cell_22/add:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_1ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_1StridedSlice%lstm_cell_22/ReadVariableOp_1:value:0+lstm_cell_22/strided_slice_1/stack:output:0-lstm_cell_22/strided_slice_1/stack_1:output:0-lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_5MatMullstm_cell_22/mul_5:z:0%lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_1AddV2lstm_cell_22/BiasAdd_1:output:0lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         y
lstm_cell_22/mul_8Mullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_2ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_2StridedSlice%lstm_cell_22/ReadVariableOp_2:value:0+lstm_cell_22/strided_slice_2/stack:output:0-lstm_cell_22/strided_slice_2/stack_1:output:0-lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_6MatMullstm_cell_22/mul_6:z:0%lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_2AddV2lstm_cell_22/BiasAdd_2:output:0lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell_22/mul_9Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         }
lstm_cell_22/add_3AddV2lstm_cell_22/mul_8:z:0lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_3ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_3StridedSlice%lstm_cell_22/ReadVariableOp_3:value:0+lstm_cell_22/strided_slice_3/stack:output:0-lstm_cell_22/strided_slice_3/stack_1:output:0-lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_7MatMullstm_cell_22/mul_7:z:0%lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_4AddV2lstm_cell_22/BiasAdd_3:output:0lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_3Sigmoidlstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_4Sigmoidlstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mul_10Mullstm_cell_22/Sigmoid_3:y:0lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_22_split_readvariableop_resource,lstm_cell_22_split_1_readvariableop_resource$lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_441787*
condR
while_cond_441786*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┬
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp^lstm_cell_22/ReadVariableOp^lstm_cell_22/ReadVariableOp_1^lstm_cell_22/ReadVariableOp_2^lstm_cell_22/ReadVariableOp_3"^lstm_cell_22/split/ReadVariableOp$^lstm_cell_22/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 2:
lstm_cell_22/ReadVariableOplstm_cell_22/ReadVariableOp2>
lstm_cell_22/ReadVariableOp_1lstm_cell_22/ReadVariableOp_12>
lstm_cell_22/ReadVariableOp_2lstm_cell_22/ReadVariableOp_22>
lstm_cell_22/ReadVariableOp_3lstm_cell_22/ReadVariableOp_32F
!lstm_cell_22/split/ReadVariableOp!lstm_cell_22/split/ReadVariableOp2J
#lstm_cell_22/split_1/ReadVariableOp#lstm_cell_22/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
ъ
ч
'sequential_21_lstm_22_while_cond_440573H
Dsequential_21_lstm_22_while_sequential_21_lstm_22_while_loop_counterN
Jsequential_21_lstm_22_while_sequential_21_lstm_22_while_maximum_iterations+
'sequential_21_lstm_22_while_placeholder-
)sequential_21_lstm_22_while_placeholder_1-
)sequential_21_lstm_22_while_placeholder_2-
)sequential_21_lstm_22_while_placeholder_3J
Fsequential_21_lstm_22_while_less_sequential_21_lstm_22_strided_slice_1`
\sequential_21_lstm_22_while_sequential_21_lstm_22_while_cond_440573___redundant_placeholder0`
\sequential_21_lstm_22_while_sequential_21_lstm_22_while_cond_440573___redundant_placeholder1`
\sequential_21_lstm_22_while_sequential_21_lstm_22_while_cond_440573___redundant_placeholder2`
\sequential_21_lstm_22_while_sequential_21_lstm_22_while_cond_440573___redundant_placeholder3(
$sequential_21_lstm_22_while_identity
║
 sequential_21/lstm_22/while/LessLess'sequential_21_lstm_22_while_placeholderFsequential_21_lstm_22_while_less_sequential_21_lstm_22_strided_slice_1*
T0*
_output_shapes
: w
$sequential_21/lstm_22/while/IdentityIdentity$sequential_21/lstm_22/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_21_lstm_22_while_identity-sequential_21/lstm_22/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
н─
ъ	
while_body_441787
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_22_split_readvariableop_resource_0:B
4while_lstm_cell_22_split_1_readvariableop_resource_0:>
,while_lstm_cell_22_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_22_split_readvariableop_resource:@
2while_lstm_cell_22_split_1_readvariableop_resource:<
*while_lstm_cell_22_readvariableop_resource:ѕб!while/lstm_cell_22/ReadVariableOpб#while/lstm_cell_22/ReadVariableOp_1б#while/lstm_cell_22/ReadVariableOp_2б#while/lstm_cell_22/ReadVariableOp_3б'while/lstm_cell_22/split/ReadVariableOpб)while/lstm_cell_22/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ѓ
"while/lstm_cell_22/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?░
while/lstm_cell_22/ones_likeFill+while/lstm_cell_22/ones_like/Shape:output:0+while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         e
 while/lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Е
while/lstm_cell_22/dropout/MulMul%while/lstm_cell_22/ones_like:output:0)while/lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         u
 while/lstm_cell_22/dropout/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:▓
7while/lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0n
)while/lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=▀
'while/lstm_cell_22/dropout/GreaterEqualGreaterEqual@while/lstm_cell_22/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ћ
while/lstm_cell_22/dropout/CastCast+while/lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         б
 while/lstm_cell_22/dropout/Mul_1Mul"while/lstm_cell_22/dropout/Mul:z:0#while/lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_1/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_1/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_1/CastCast-while/lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_1/Mul_1Mul$while/lstm_cell_22/dropout_1/Mul:z:0%while/lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_2/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_2/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_2/CastCast-while/lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_2/Mul_1Mul$while/lstm_cell_22/dropout_2/Mul:z:0%while/lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Г
 while/lstm_cell_22/dropout_3/MulMul%while/lstm_cell_22/ones_like:output:0+while/lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         w
"while/lstm_cell_22/dropout_3/ShapeShape%while/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_3/CastCast-while/lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_3/Mul_1Mul$while/lstm_cell_22/dropout_3/Mul:z:0%while/lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         g
$while/lstm_cell_22/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
while/lstm_cell_22/ones_like_1Fill-while/lstm_cell_22/ones_like_1/Shape:output:0-while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_4/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_4/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_4/CastCast-while/lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_4/Mul_1Mul$while/lstm_cell_22/dropout_4/Mul:z:0%while/lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_5/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_5/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_5/CastCast-while/lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_5/Mul_1Mul$while/lstm_cell_22/dropout_5/Mul:z:0%while/lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_6/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_6/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_6/CastCast-while/lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_6/Mul_1Mul$while/lstm_cell_22/dropout_6/Mul:z:0%while/lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         g
"while/lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 while/lstm_cell_22/dropout_7/MulMul'while/lstm_cell_22/ones_like_1:output:0+while/lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         y
"while/lstm_cell_22/dropout_7/ShapeShape'while/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:Х
9while/lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+while/lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)while/lstm_cell_22/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!while/lstm_cell_22/dropout_7/CastCast-while/lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"while/lstm_cell_22/dropout_7/Mul_1Mul$while/lstm_cell_22/dropout_7/Mul:z:0%while/lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         Д
while/lstm_cell_22/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         Ф
while/lstm_cell_22/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :џ
'while/lstm_cell_22/split/ReadVariableOpReadVariableOp2while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0О
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitћ
while/lstm_cell_22/MatMulMatMulwhile/lstm_cell_22/mul:z:0!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_1MatMulwhile/lstm_cell_22/mul_1:z:0!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_2MatMulwhile/lstm_cell_22/mul_2:z:0!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_3MatMulwhile/lstm_cell_22/mul_3:z:0!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         f
$while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : џ
)while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0═
while/lstm_cell_22/split_1Split-while/lstm_cell_22/split_1/split_dim:output:01while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitА
while/lstm_cell_22/BiasAddBiasAdd#while/lstm_cell_22/MatMul:product:0#while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_1BiasAdd%while/lstm_cell_22/MatMul_1:product:0#while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_2BiasAdd%while/lstm_cell_22/MatMul_2:product:0#while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_3BiasAdd%while/lstm_cell_22/MatMul_3:product:0#while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_4Mulwhile_placeholder_2&while/lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_5Mulwhile_placeholder_2&while/lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_6Mulwhile_placeholder_2&while/lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         ј
while/lstm_cell_22/mul_7Mulwhile_placeholder_2&while/lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         ј
!while/lstm_cell_22/ReadVariableOpReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_22/strided_sliceStridedSlice)while/lstm_cell_22/ReadVariableOp:value:0/while/lstm_cell_22/strided_slice/stack:output:01while/lstm_cell_22/strided_slice/stack_1:output:01while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskа
while/lstm_cell_22/MatMul_4MatMulwhile/lstm_cell_22/mul_4:z:0)while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/BiasAdd:output:0%while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         s
while/lstm_cell_22/SigmoidSigmoidwhile/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_1ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_1StridedSlice+while/lstm_cell_22/ReadVariableOp_1:value:01while/lstm_cell_22/strided_slice_1/stack:output:03while/lstm_cell_22/strided_slice_1/stack_1:output:03while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_5MatMulwhile/lstm_cell_22/mul_5:z:0+while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_1AddV2%while/lstm_cell_22/BiasAdd_1:output:0%while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_1Sigmoidwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ѕ
while/lstm_cell_22/mul_8Mul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_2ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_2StridedSlice+while/lstm_cell_22/ReadVariableOp_2:value:01while/lstm_cell_22/strided_slice_2/stack:output:03while/lstm_cell_22/strided_slice_2/stack_1:output:03while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_6MatMulwhile/lstm_cell_22/mul_6:z:0+while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_2AddV2%while/lstm_cell_22/BiasAdd_2:output:0%while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_2Sigmoidwhile/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell_22/mul_9Mulwhile/lstm_cell_22/Sigmoid:y:0 while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/add_3AddV2while/lstm_cell_22/mul_8:z:0while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_3ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_3StridedSlice+while/lstm_cell_22/ReadVariableOp_3:value:01while/lstm_cell_22/strided_slice_3/stack:output:03while/lstm_cell_22/strided_slice_3/stack_1:output:03while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_7MatMulwhile/lstm_cell_22/mul_7:z:0+while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_4AddV2%while/lstm_cell_22/BiasAdd_3:output:0%while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_3Sigmoidwhile/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_4Sigmoidwhile/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ќ
while/lstm_cell_22/mul_10Mul while/lstm_cell_22/Sigmoid_3:y:0 while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         к
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмz
while/Identity_4Identitywhile/lstm_cell_22/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         И

while/NoOpNoOp"^while/lstm_cell_22/ReadVariableOp$^while/lstm_cell_22/ReadVariableOp_1$^while/lstm_cell_22/ReadVariableOp_2$^while/lstm_cell_22/ReadVariableOp_3(^while/lstm_cell_22/split/ReadVariableOp*^while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_22_readvariableop_resource,while_lstm_cell_22_readvariableop_resource_0"j
2while_lstm_cell_22_split_1_readvariableop_resource4while_lstm_cell_22_split_1_readvariableop_resource_0"f
0while_lstm_cell_22_split_readvariableop_resource2while_lstm_cell_22_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2F
!while/lstm_cell_22/ReadVariableOp!while/lstm_cell_22/ReadVariableOp2J
#while/lstm_cell_22/ReadVariableOp_1#while/lstm_cell_22/ReadVariableOp_12J
#while/lstm_cell_22/ReadVariableOp_2#while/lstm_cell_22/ReadVariableOp_22J
#while/lstm_cell_22/ReadVariableOp_3#while/lstm_cell_22/ReadVariableOp_32R
'while/lstm_cell_22/split/ReadVariableOp'while/lstm_cell_22/split/ReadVariableOp2V
)while/lstm_cell_22/split_1/ReadVariableOp)while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
ёv
ъ	
while_body_443724
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_22_split_readvariableop_resource_0:B
4while_lstm_cell_22_split_1_readvariableop_resource_0:>
,while_lstm_cell_22_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_22_split_readvariableop_resource:@
2while_lstm_cell_22_split_1_readvariableop_resource:<
*while_lstm_cell_22_readvariableop_resource:ѕб!while/lstm_cell_22/ReadVariableOpб#while/lstm_cell_22/ReadVariableOp_1б#while/lstm_cell_22/ReadVariableOp_2б#while/lstm_cell_22/ReadVariableOp_3б'while/lstm_cell_22/split/ReadVariableOpб)while/lstm_cell_22/split_1/ReadVariableOpѕ
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       д
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:         *
element_dtype0ѓ
"while/lstm_cell_22/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?░
while/lstm_cell_22/ones_likeFill+while/lstm_cell_22/ones_like/Shape:output:0+while/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         g
$while/lstm_cell_22/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
while/lstm_cell_22/ones_like_1Fill-while/lstm_cell_22/ones_like_1/Shape:output:0-while/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         е
while/lstm_cell_22/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         ф
while/lstm_cell_22/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_22/ones_like:output:0*
T0*'
_output_shapes
:         d
"while/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :џ
'while/lstm_cell_22/split/ReadVariableOpReadVariableOp2while_lstm_cell_22_split_readvariableop_resource_0*
_output_shapes

:*
dtype0О
while/lstm_cell_22/splitSplit+while/lstm_cell_22/split/split_dim:output:0/while/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitћ
while/lstm_cell_22/MatMulMatMulwhile/lstm_cell_22/mul:z:0!while/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_1MatMulwhile/lstm_cell_22/mul_1:z:0!while/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_2MatMulwhile/lstm_cell_22/mul_2:z:0!while/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ў
while/lstm_cell_22/MatMul_3MatMulwhile/lstm_cell_22/mul_3:z:0!while/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         f
$while/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : џ
)while/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_22_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0═
while/lstm_cell_22/split_1Split-while/lstm_cell_22/split_1/split_dim:output:01while/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitА
while/lstm_cell_22/BiasAddBiasAdd#while/lstm_cell_22/MatMul:product:0#while/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_1BiasAdd%while/lstm_cell_22/MatMul_1:product:0#while/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_2BiasAdd%while/lstm_cell_22/MatMul_2:product:0#while/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ц
while/lstm_cell_22/BiasAdd_3BiasAdd%while/lstm_cell_22/MatMul_3:product:0#while/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_4Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_5Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_6Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/mul_7Mulwhile_placeholder_2'while/lstm_cell_22/ones_like_1:output:0*
T0*'
_output_shapes
:         ј
!while/lstm_cell_22/ReadVariableOpReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ╩
 while/lstm_cell_22/strided_sliceStridedSlice)while/lstm_cell_22/ReadVariableOp:value:0/while/lstm_cell_22/strided_slice/stack:output:01while/lstm_cell_22/strided_slice/stack_1:output:01while/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskа
while/lstm_cell_22/MatMul_4MatMulwhile/lstm_cell_22/mul_4:z:0)while/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Ю
while/lstm_cell_22/addAddV2#while/lstm_cell_22/BiasAdd:output:0%while/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         s
while/lstm_cell_22/SigmoidSigmoidwhile/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_1ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_1StridedSlice+while/lstm_cell_22/ReadVariableOp_1:value:01while/lstm_cell_22/strided_slice_1/stack:output:03while/lstm_cell_22/strided_slice_1/stack_1:output:03while/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_5MatMulwhile/lstm_cell_22/mul_5:z:0+while/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_1AddV2%while/lstm_cell_22/BiasAdd_1:output:0%while/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_1Sigmoidwhile/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         ѕ
while/lstm_cell_22/mul_8Mul while/lstm_cell_22/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_2ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_2StridedSlice+while/lstm_cell_22/ReadVariableOp_2:value:01while/lstm_cell_22/strided_slice_2/stack:output:03while/lstm_cell_22/strided_slice_2/stack_1:output:03while/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_6MatMulwhile/lstm_cell_22/mul_6:z:0+while/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_2AddV2%while/lstm_cell_22/BiasAdd_2:output:0%while/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_2Sigmoidwhile/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Њ
while/lstm_cell_22/mul_9Mulwhile/lstm_cell_22/Sigmoid:y:0 while/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ј
while/lstm_cell_22/add_3AddV2while/lstm_cell_22/mul_8:z:0while/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         љ
#while/lstm_cell_22/ReadVariableOp_3ReadVariableOp,while_lstm_cell_22_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"while/lstm_cell_22/strided_slice_3StridedSlice+while/lstm_cell_22/ReadVariableOp_3:value:01while/lstm_cell_22/strided_slice_3/stack:output:03while/lstm_cell_22/strided_slice_3/stack_1:output:03while/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskб
while/lstm_cell_22/MatMul_7MatMulwhile/lstm_cell_22/mul_7:z:0+while/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         А
while/lstm_cell_22/add_4AddV2%while/lstm_cell_22/BiasAdd_3:output:0%while/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_3Sigmoidwhile/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         w
while/lstm_cell_22/Sigmoid_4Sigmoidwhile/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ќ
while/lstm_cell_22/mul_10Mul while/lstm_cell_22/Sigmoid_3:y:0 while/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         к
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_22/mul_10:z:0*
_output_shapes
: *
element_dtype0:жУмM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: Ў
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :жУмz
while/Identity_4Identitywhile/lstm_cell_22/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:         y
while/Identity_5Identitywhile/lstm_cell_22/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:         И

while/NoOpNoOp"^while/lstm_cell_22/ReadVariableOp$^while/lstm_cell_22/ReadVariableOp_1$^while/lstm_cell_22/ReadVariableOp_2$^while/lstm_cell_22/ReadVariableOp_3(^while/lstm_cell_22/split/ReadVariableOp*^while/lstm_cell_22/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_22_readvariableop_resource,while_lstm_cell_22_readvariableop_resource_0"j
2while_lstm_cell_22_split_1_readvariableop_resource4while_lstm_cell_22_split_1_readvariableop_resource_0"f
0while_lstm_cell_22_split_readvariableop_resource2while_lstm_cell_22_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"е
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :         :         : : : : : 2F
!while/lstm_cell_22/ReadVariableOp!while/lstm_cell_22/ReadVariableOp2J
#while/lstm_cell_22/ReadVariableOp_1#while/lstm_cell_22/ReadVariableOp_12J
#while/lstm_cell_22/ReadVariableOp_2#while/lstm_cell_22/ReadVariableOp_22J
#while/lstm_cell_22/ReadVariableOp_3#while/lstm_cell_22/ReadVariableOp_32R
'while/lstm_cell_22/split/ReadVariableOp'while/lstm_cell_22/split/ReadVariableOp2V
)while/lstm_cell_22/split_1/ReadVariableOp)while/lstm_cell_22/split_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
: 
х
├
while_cond_444030
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_444030___redundant_placeholder04
0while_while_cond_444030___redundant_placeholder14
0while_while_cond_444030___redundant_placeholder24
0while_while_cond_444030___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :         :         : ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:         :-)
'
_output_shapes
:         :

_output_shapes
: :

_output_shapes
:
І8
ѓ
C__inference_lstm_22_layer_call_and_return_conditional_losses_441225

inputs%
lstm_cell_22_441143:!
lstm_cell_22_441145:%
lstm_cell_22_441147:
identityѕб$lstm_cell_22/StatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskш
$lstm_cell_22/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_22_441143lstm_cell_22_441145lstm_cell_22_441147*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:         :         :         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *Q
fLRJ
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_441097n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : и
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_22_441143lstm_cell_22_441145lstm_cell_22_441147*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_441156*
condR
while_cond_441155*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         u
NoOpNoOp%^lstm_cell_22/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2L
$lstm_cell_22/StatefulPartitionedCall$lstm_cell_22/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
Рљ
Р
I__inference_sequential_21_layer_call_and_return_conditional_losses_442893

inputs<
*dense_66_tensordot_readvariableop_resource:6
(dense_66_biasadd_readvariableop_resource:D
2lstm_22_lstm_cell_22_split_readvariableop_resource:B
4lstm_22_lstm_cell_22_split_1_readvariableop_resource:>
,lstm_22_lstm_cell_22_readvariableop_resource:9
'dense_67_matmul_readvariableop_resource:6
(dense_67_biasadd_readvariableop_resource:9
'dense_68_matmul_readvariableop_resource:6
(dense_68_biasadd_readvariableop_resource:
identityѕбdense_66/BiasAdd/ReadVariableOpб!dense_66/Tensordot/ReadVariableOpбdense_67/BiasAdd/ReadVariableOpбdense_67/MatMul/ReadVariableOpбdense_68/BiasAdd/ReadVariableOpбdense_68/MatMul/ReadVariableOpб#lstm_22/lstm_cell_22/ReadVariableOpб%lstm_22/lstm_cell_22/ReadVariableOp_1б%lstm_22/lstm_cell_22/ReadVariableOp_2б%lstm_22/lstm_cell_22/ReadVariableOp_3б)lstm_22/lstm_cell_22/split/ReadVariableOpб+lstm_22/lstm_cell_22/split_1/ReadVariableOpбlstm_22/whileї
!dense_66/Tensordot/ReadVariableOpReadVariableOp*dense_66_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_66/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_66/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_66/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_66/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
dense_66/Tensordot/GatherV2GatherV2!dense_66/Tensordot/Shape:output:0 dense_66/Tensordot/free:output:0)dense_66/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_66/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : с
dense_66/Tensordot/GatherV2_1GatherV2!dense_66/Tensordot/Shape:output:0 dense_66/Tensordot/axes:output:0+dense_66/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_66/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Ѕ
dense_66/Tensordot/ProdProd$dense_66/Tensordot/GatherV2:output:0!dense_66/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_66/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ј
dense_66/Tensordot/Prod_1Prod&dense_66/Tensordot/GatherV2_1:output:0#dense_66/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_66/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : └
dense_66/Tensordot/concatConcatV2 dense_66/Tensordot/free:output:0 dense_66/Tensordot/axes:output:0'dense_66/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:ћ
dense_66/Tensordot/stackPack dense_66/Tensordot/Prod:output:0"dense_66/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:І
dense_66/Tensordot/transpose	Transposeinputs"dense_66/Tensordot/concat:output:0*
T0*+
_output_shapes
:         Ц
dense_66/Tensordot/ReshapeReshape dense_66/Tensordot/transpose:y:0!dense_66/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  Ц
dense_66/Tensordot/MatMulMatMul#dense_66/Tensordot/Reshape:output:0)dense_66/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         d
dense_66/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_66/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╦
dense_66/Tensordot/concat_1ConcatV2$dense_66/Tensordot/GatherV2:output:0#dense_66/Tensordot/Const_2:output:0)dense_66/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:ъ
dense_66/TensordotReshape#dense_66/Tensordot/MatMul:product:0$dense_66/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         ё
dense_66/BiasAdd/ReadVariableOpReadVariableOp(dense_66_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ќ
dense_66/BiasAddBiasAdddense_66/Tensordot:output:0'dense_66/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         V
lstm_22/ShapeShapedense_66/BiasAdd:output:0*
T0*
_output_shapes
:e
lstm_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:щ
lstm_22/strided_sliceStridedSlicelstm_22/Shape:output:0$lstm_22/strided_slice/stack:output:0&lstm_22/strided_slice/stack_1:output:0&lstm_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_22/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :І
lstm_22/zeros/packedPacklstm_22/strided_slice:output:0lstm_22/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_22/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ё
lstm_22/zerosFilllstm_22/zeros/packed:output:0lstm_22/zeros/Const:output:0*
T0*'
_output_shapes
:         Z
lstm_22/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :Ј
lstm_22/zeros_1/packedPacklstm_22/strided_slice:output:0!lstm_22/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_22/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    і
lstm_22/zeros_1Filllstm_22/zeros_1/packed:output:0lstm_22/zeros_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_22/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          љ
lstm_22/transpose	Transposedense_66/BiasAdd:output:0lstm_22/transpose/perm:output:0*
T0*+
_output_shapes
:         T
lstm_22/Shape_1Shapelstm_22/transpose:y:0*
T0*
_output_shapes
:g
lstm_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ѓ
lstm_22/strided_slice_1StridedSlicelstm_22/Shape_1:output:0&lstm_22/strided_slice_1/stack:output:0(lstm_22/strided_slice_1/stack_1:output:0(lstm_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_22/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ╠
lstm_22/TensorArrayV2TensorListReserve,lstm_22/TensorArrayV2/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмј
=lstm_22/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Э
/lstm_22/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_22/transpose:y:0Flstm_22/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмg
lstm_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Љ
lstm_22/strided_slice_2StridedSlicelstm_22/transpose:y:0&lstm_22/strided_slice_2/stack:output:0(lstm_22/strided_slice_2/stack_1:output:0(lstm_22/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskt
$lstm_22/lstm_cell_22/ones_like/ShapeShape lstm_22/strided_slice_2:output:0*
T0*
_output_shapes
:i
$lstm_22/lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?Х
lstm_22/lstm_cell_22/ones_likeFill-lstm_22/lstm_cell_22/ones_like/Shape:output:0-lstm_22/lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         g
"lstm_22/lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?»
 lstm_22/lstm_cell_22/dropout/MulMul'lstm_22/lstm_cell_22/ones_like:output:0+lstm_22/lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         y
"lstm_22/lstm_cell_22/dropout/ShapeShape'lstm_22/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:Х
9lstm_22/lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform+lstm_22/lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0p
+lstm_22/lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=т
)lstm_22/lstm_cell_22/dropout/GreaterEqualGreaterEqualBlstm_22/lstm_cell_22/dropout/random_uniform/RandomUniform:output:04lstm_22/lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ў
!lstm_22/lstm_cell_22/dropout/CastCast-lstm_22/lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         е
"lstm_22/lstm_cell_22/dropout/Mul_1Mul$lstm_22/lstm_cell_22/dropout/Mul:z:0%lstm_22/lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         i
$lstm_22/lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?│
"lstm_22/lstm_cell_22/dropout_1/MulMul'lstm_22/lstm_cell_22/ones_like:output:0-lstm_22/lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         {
$lstm_22/lstm_cell_22/dropout_1/ShapeShape'lstm_22/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:║
;lstm_22/lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform-lstm_22/lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0r
-lstm_22/lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=в
+lstm_22/lstm_cell_22/dropout_1/GreaterEqualGreaterEqualDlstm_22/lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:06lstm_22/lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ю
#lstm_22/lstm_cell_22/dropout_1/CastCast/lstm_22/lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         «
$lstm_22/lstm_cell_22/dropout_1/Mul_1Mul&lstm_22/lstm_cell_22/dropout_1/Mul:z:0'lstm_22/lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         i
$lstm_22/lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?│
"lstm_22/lstm_cell_22/dropout_2/MulMul'lstm_22/lstm_cell_22/ones_like:output:0-lstm_22/lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         {
$lstm_22/lstm_cell_22/dropout_2/ShapeShape'lstm_22/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:║
;lstm_22/lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform-lstm_22/lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0r
-lstm_22/lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=в
+lstm_22/lstm_cell_22/dropout_2/GreaterEqualGreaterEqualDlstm_22/lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:06lstm_22/lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ю
#lstm_22/lstm_cell_22/dropout_2/CastCast/lstm_22/lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         «
$lstm_22/lstm_cell_22/dropout_2/Mul_1Mul&lstm_22/lstm_cell_22/dropout_2/Mul:z:0'lstm_22/lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         i
$lstm_22/lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?│
"lstm_22/lstm_cell_22/dropout_3/MulMul'lstm_22/lstm_cell_22/ones_like:output:0-lstm_22/lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         {
$lstm_22/lstm_cell_22/dropout_3/ShapeShape'lstm_22/lstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:║
;lstm_22/lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform-lstm_22/lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0r
-lstm_22/lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=в
+lstm_22/lstm_cell_22/dropout_3/GreaterEqualGreaterEqualDlstm_22/lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:06lstm_22/lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ю
#lstm_22/lstm_cell_22/dropout_3/CastCast/lstm_22/lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         «
$lstm_22/lstm_cell_22/dropout_3/Mul_1Mul&lstm_22/lstm_cell_22/dropout_3/Mul:z:0'lstm_22/lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         l
&lstm_22/lstm_cell_22/ones_like_1/ShapeShapelstm_22/zeros:output:0*
T0*
_output_shapes
:k
&lstm_22/lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?╝
 lstm_22/lstm_cell_22/ones_like_1Fill/lstm_22/lstm_cell_22/ones_like_1/Shape:output:0/lstm_22/lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         i
$lstm_22/lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?х
"lstm_22/lstm_cell_22/dropout_4/MulMul)lstm_22/lstm_cell_22/ones_like_1:output:0-lstm_22/lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         }
$lstm_22/lstm_cell_22/dropout_4/ShapeShape)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:║
;lstm_22/lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform-lstm_22/lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0r
-lstm_22/lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=в
+lstm_22/lstm_cell_22/dropout_4/GreaterEqualGreaterEqualDlstm_22/lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:06lstm_22/lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ю
#lstm_22/lstm_cell_22/dropout_4/CastCast/lstm_22/lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         «
$lstm_22/lstm_cell_22/dropout_4/Mul_1Mul&lstm_22/lstm_cell_22/dropout_4/Mul:z:0'lstm_22/lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         i
$lstm_22/lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?х
"lstm_22/lstm_cell_22/dropout_5/MulMul)lstm_22/lstm_cell_22/ones_like_1:output:0-lstm_22/lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         }
$lstm_22/lstm_cell_22/dropout_5/ShapeShape)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:║
;lstm_22/lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform-lstm_22/lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0r
-lstm_22/lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=в
+lstm_22/lstm_cell_22/dropout_5/GreaterEqualGreaterEqualDlstm_22/lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:06lstm_22/lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ю
#lstm_22/lstm_cell_22/dropout_5/CastCast/lstm_22/lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         «
$lstm_22/lstm_cell_22/dropout_5/Mul_1Mul&lstm_22/lstm_cell_22/dropout_5/Mul:z:0'lstm_22/lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         i
$lstm_22/lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?х
"lstm_22/lstm_cell_22/dropout_6/MulMul)lstm_22/lstm_cell_22/ones_like_1:output:0-lstm_22/lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         }
$lstm_22/lstm_cell_22/dropout_6/ShapeShape)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:║
;lstm_22/lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform-lstm_22/lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0r
-lstm_22/lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=в
+lstm_22/lstm_cell_22/dropout_6/GreaterEqualGreaterEqualDlstm_22/lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:06lstm_22/lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ю
#lstm_22/lstm_cell_22/dropout_6/CastCast/lstm_22/lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         «
$lstm_22/lstm_cell_22/dropout_6/Mul_1Mul&lstm_22/lstm_cell_22/dropout_6/Mul:z:0'lstm_22/lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         i
$lstm_22/lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?х
"lstm_22/lstm_cell_22/dropout_7/MulMul)lstm_22/lstm_cell_22/ones_like_1:output:0-lstm_22/lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         }
$lstm_22/lstm_cell_22/dropout_7/ShapeShape)lstm_22/lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:║
;lstm_22/lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform-lstm_22/lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0r
-lstm_22/lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=в
+lstm_22/lstm_cell_22/dropout_7/GreaterEqualGreaterEqualDlstm_22/lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:06lstm_22/lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ю
#lstm_22/lstm_cell_22/dropout_7/CastCast/lstm_22/lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         «
$lstm_22/lstm_cell_22/dropout_7/Mul_1Mul&lstm_22/lstm_cell_22/dropout_7/Mul:z:0'lstm_22/lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         Џ
lstm_22/lstm_cell_22/mulMul lstm_22/strided_slice_2:output:0&lstm_22/lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Ъ
lstm_22/lstm_cell_22/mul_1Mul lstm_22/strided_slice_2:output:0(lstm_22/lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         Ъ
lstm_22/lstm_cell_22/mul_2Mul lstm_22/strided_slice_2:output:0(lstm_22/lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         Ъ
lstm_22/lstm_cell_22/mul_3Mul lstm_22/strided_slice_2:output:0(lstm_22/lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         f
$lstm_22/lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ю
)lstm_22/lstm_cell_22/split/ReadVariableOpReadVariableOp2lstm_22_lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0П
lstm_22/lstm_cell_22/splitSplit-lstm_22/lstm_cell_22/split/split_dim:output:01lstm_22/lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitџ
lstm_22/lstm_cell_22/MatMulMatMullstm_22/lstm_cell_22/mul:z:0#lstm_22/lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/MatMul_1MatMullstm_22/lstm_cell_22/mul_1:z:0#lstm_22/lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/MatMul_2MatMullstm_22/lstm_cell_22/mul_2:z:0#lstm_22/lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         ъ
lstm_22/lstm_cell_22/MatMul_3MatMullstm_22/lstm_cell_22/mul_3:z:0#lstm_22/lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         h
&lstm_22/lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ю
+lstm_22/lstm_cell_22/split_1/ReadVariableOpReadVariableOp4lstm_22_lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0М
lstm_22/lstm_cell_22/split_1Split/lstm_22/lstm_cell_22/split_1/split_dim:output:03lstm_22/lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitД
lstm_22/lstm_cell_22/BiasAddBiasAdd%lstm_22/lstm_cell_22/MatMul:product:0%lstm_22/lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Ф
lstm_22/lstm_cell_22/BiasAdd_1BiasAdd'lstm_22/lstm_cell_22/MatMul_1:product:0%lstm_22/lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Ф
lstm_22/lstm_cell_22/BiasAdd_2BiasAdd'lstm_22/lstm_cell_22/MatMul_2:product:0%lstm_22/lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Ф
lstm_22/lstm_cell_22/BiasAdd_3BiasAdd'lstm_22/lstm_cell_22/MatMul_3:product:0%lstm_22/lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         Ћ
lstm_22/lstm_cell_22/mul_4Mullstm_22/zeros:output:0(lstm_22/lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         Ћ
lstm_22/lstm_cell_22/mul_5Mullstm_22/zeros:output:0(lstm_22/lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         Ћ
lstm_22/lstm_cell_22/mul_6Mullstm_22/zeros:output:0(lstm_22/lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         Ћ
lstm_22/lstm_cell_22/mul_7Mullstm_22/zeros:output:0(lstm_22/lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         љ
#lstm_22/lstm_cell_22/ReadVariableOpReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0y
(lstm_22/lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_22/lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_22/lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      н
"lstm_22/lstm_cell_22/strided_sliceStridedSlice+lstm_22/lstm_cell_22/ReadVariableOp:value:01lstm_22/lstm_cell_22/strided_slice/stack:output:03lstm_22/lstm_cell_22/strided_slice/stack_1:output:03lstm_22/lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskд
lstm_22/lstm_cell_22/MatMul_4MatMullstm_22/lstm_cell_22/mul_4:z:0+lstm_22/lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         Б
lstm_22/lstm_cell_22/addAddV2%lstm_22/lstm_cell_22/BiasAdd:output:0'lstm_22/lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         w
lstm_22/lstm_cell_22/SigmoidSigmoidlstm_22/lstm_cell_22/add:z:0*
T0*'
_output_shapes
:         њ
%lstm_22/lstm_cell_22/ReadVariableOp_1ReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_22/lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      я
$lstm_22/lstm_cell_22/strided_slice_1StridedSlice-lstm_22/lstm_cell_22/ReadVariableOp_1:value:03lstm_22/lstm_cell_22/strided_slice_1/stack:output:05lstm_22/lstm_cell_22/strided_slice_1/stack_1:output:05lstm_22/lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskе
lstm_22/lstm_cell_22/MatMul_5MatMullstm_22/lstm_cell_22/mul_5:z:0-lstm_22/lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Д
lstm_22/lstm_cell_22/add_1AddV2'lstm_22/lstm_cell_22/BiasAdd_1:output:0'lstm_22/lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_1Sigmoidlstm_22/lstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         Љ
lstm_22/lstm_cell_22/mul_8Mul"lstm_22/lstm_cell_22/Sigmoid_1:y:0lstm_22/zeros_1:output:0*
T0*'
_output_shapes
:         њ
%lstm_22/lstm_cell_22/ReadVariableOp_2ReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_22/lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      я
$lstm_22/lstm_cell_22/strided_slice_2StridedSlice-lstm_22/lstm_cell_22/ReadVariableOp_2:value:03lstm_22/lstm_cell_22/strided_slice_2/stack:output:05lstm_22/lstm_cell_22/strided_slice_2/stack_1:output:05lstm_22/lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskе
lstm_22/lstm_cell_22/MatMul_6MatMullstm_22/lstm_cell_22/mul_6:z:0-lstm_22/lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Д
lstm_22/lstm_cell_22/add_2AddV2'lstm_22/lstm_cell_22/BiasAdd_2:output:0'lstm_22/lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_2Sigmoidlstm_22/lstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ў
lstm_22/lstm_cell_22/mul_9Mul lstm_22/lstm_cell_22/Sigmoid:y:0"lstm_22/lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         Ћ
lstm_22/lstm_cell_22/add_3AddV2lstm_22/lstm_cell_22/mul_8:z:0lstm_22/lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         њ
%lstm_22/lstm_cell_22/ReadVariableOp_3ReadVariableOp,lstm_22_lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_22/lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_22/lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_22/lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      я
$lstm_22/lstm_cell_22/strided_slice_3StridedSlice-lstm_22/lstm_cell_22/ReadVariableOp_3:value:03lstm_22/lstm_cell_22/strided_slice_3/stack:output:05lstm_22/lstm_cell_22/strided_slice_3/stack_1:output:05lstm_22/lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskе
lstm_22/lstm_cell_22/MatMul_7MatMullstm_22/lstm_cell_22/mul_7:z:0-lstm_22/lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Д
lstm_22/lstm_cell_22/add_4AddV2'lstm_22/lstm_cell_22/BiasAdd_3:output:0'lstm_22/lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_3Sigmoidlstm_22/lstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         {
lstm_22/lstm_cell_22/Sigmoid_4Sigmoidlstm_22/lstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ю
lstm_22/lstm_cell_22/mul_10Mul"lstm_22/lstm_cell_22/Sigmoid_3:y:0"lstm_22/lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         v
%lstm_22/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       л
lstm_22/TensorArrayV2_1TensorListReserve.lstm_22/TensorArrayV2_1/element_shape:output:0 lstm_22/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмN
lstm_22/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_22/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         \
lstm_22/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : У
lstm_22/whileWhile#lstm_22/while/loop_counter:output:0)lstm_22/while/maximum_iterations:output:0lstm_22/time:output:0 lstm_22/TensorArrayV2_1:handle:0lstm_22/zeros:output:0lstm_22/zeros_1:output:0 lstm_22/strided_slice_1:output:0?lstm_22/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_22_lstm_cell_22_split_readvariableop_resource4lstm_22_lstm_cell_22_split_1_readvariableop_resource,lstm_22_lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_22_while_body_442683*%
condR
lstm_22_while_cond_442682*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ѕ
8lstm_22/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ┌
*lstm_22/TensorArrayV2Stack/TensorListStackTensorListStacklstm_22/while:output:3Alstm_22/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:         *
element_dtype0p
lstm_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         i
lstm_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:»
lstm_22/strided_slice_3StridedSlice3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_22/strided_slice_3/stack:output:0(lstm_22/strided_slice_3/stack_1:output:0(lstm_22/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskm
lstm_22/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_22/transpose_1	Transpose3lstm_22/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_22/transpose_1/perm:output:0*
T0*+
_output_shapes
:         c
lstm_22/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    є
dense_67/MatMul/ReadVariableOpReadVariableOp'dense_67_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ћ
dense_67/MatMulMatMul lstm_22/strided_slice_3:output:0&dense_67/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_67/BiasAdd/ReadVariableOpReadVariableOp(dense_67_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_67/BiasAddBiasAdddense_67/MatMul:product:0'dense_67/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         є
dense_68/MatMul/ReadVariableOpReadVariableOp'dense_68_matmul_readvariableop_resource*
_output_shapes

:*
dtype0ј
dense_68/MatMulMatMuldense_67/BiasAdd:output:0&dense_68/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ё
dense_68/BiasAdd/ReadVariableOpReadVariableOp(dense_68_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
dense_68/BiasAddBiasAdddense_68/MatMul:product:0'dense_68/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         h
IdentityIdentitydense_68/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         џ
NoOpNoOp ^dense_66/BiasAdd/ReadVariableOp"^dense_66/Tensordot/ReadVariableOp ^dense_67/BiasAdd/ReadVariableOp^dense_67/MatMul/ReadVariableOp ^dense_68/BiasAdd/ReadVariableOp^dense_68/MatMul/ReadVariableOp$^lstm_22/lstm_cell_22/ReadVariableOp&^lstm_22/lstm_cell_22/ReadVariableOp_1&^lstm_22/lstm_cell_22/ReadVariableOp_2&^lstm_22/lstm_cell_22/ReadVariableOp_3*^lstm_22/lstm_cell_22/split/ReadVariableOp,^lstm_22/lstm_cell_22/split_1/ReadVariableOp^lstm_22/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):         : : : : : : : : : 2B
dense_66/BiasAdd/ReadVariableOpdense_66/BiasAdd/ReadVariableOp2F
!dense_66/Tensordot/ReadVariableOp!dense_66/Tensordot/ReadVariableOp2B
dense_67/BiasAdd/ReadVariableOpdense_67/BiasAdd/ReadVariableOp2@
dense_67/MatMul/ReadVariableOpdense_67/MatMul/ReadVariableOp2B
dense_68/BiasAdd/ReadVariableOpdense_68/BiasAdd/ReadVariableOp2@
dense_68/MatMul/ReadVariableOpdense_68/MatMul/ReadVariableOp2J
#lstm_22/lstm_cell_22/ReadVariableOp#lstm_22/lstm_cell_22/ReadVariableOp2N
%lstm_22/lstm_cell_22/ReadVariableOp_1%lstm_22/lstm_cell_22/ReadVariableOp_12N
%lstm_22/lstm_cell_22/ReadVariableOp_2%lstm_22/lstm_cell_22/ReadVariableOp_22N
%lstm_22/lstm_cell_22/ReadVariableOp_3%lstm_22/lstm_cell_22/ReadVariableOp_32V
)lstm_22/lstm_cell_22/split/ReadVariableOp)lstm_22/lstm_cell_22/split/ReadVariableOp2Z
+lstm_22/lstm_cell_22/split_1/ReadVariableOp+lstm_22/lstm_cell_22/split_1/ReadVariableOp2
lstm_22/whilelstm_22/while:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
хL
У
__inference__traced_save_444660
file_prefix.
*savev2_dense_66_kernel_read_readvariableop,
(savev2_dense_66_bias_read_readvariableop.
*savev2_dense_67_kernel_read_readvariableop,
(savev2_dense_67_bias_read_readvariableop.
*savev2_dense_68_kernel_read_readvariableop,
(savev2_dense_68_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableopD
@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop8
4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_66_kernel_m_read_readvariableop3
/savev2_adam_dense_66_bias_m_read_readvariableop5
1savev2_adam_dense_67_kernel_m_read_readvariableop3
/savev2_adam_dense_67_bias_m_read_readvariableop5
1savev2_adam_dense_68_kernel_m_read_readvariableop3
/savev2_adam_dense_68_bias_m_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_22_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_22_bias_m_read_readvariableop5
1savev2_adam_dense_66_kernel_v_read_readvariableop3
/savev2_adam_dense_66_bias_v_read_readvariableop5
1savev2_adam_dense_67_kernel_v_read_readvariableop3
/savev2_adam_dense_67_bias_v_read_readvariableop5
1savev2_adam_dense_68_kernel_v_read_readvariableop3
/savev2_adam_dense_68_bias_v_read_readvariableopA
=savev2_adam_lstm_22_lstm_cell_22_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_22_lstm_cell_22_bias_v_read_readvariableop
savev2_const

identity_1ѕбMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partЂ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ё
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*«
valueцBА%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHи
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B └
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_66_kernel_read_readvariableop(savev2_dense_66_bias_read_readvariableop*savev2_dense_67_kernel_read_readvariableop(savev2_dense_67_bias_read_readvariableop*savev2_dense_68_kernel_read_readvariableop(savev2_dense_68_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_22_lstm_cell_22_kernel_read_readvariableop@savev2_lstm_22_lstm_cell_22_recurrent_kernel_read_readvariableop4savev2_lstm_22_lstm_cell_22_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_66_kernel_m_read_readvariableop/savev2_adam_dense_66_bias_m_read_readvariableop1savev2_adam_dense_67_kernel_m_read_readvariableop/savev2_adam_dense_67_bias_m_read_readvariableop1savev2_adam_dense_68_kernel_m_read_readvariableop/savev2_adam_dense_68_bias_m_read_readvariableop=savev2_adam_lstm_22_lstm_cell_22_kernel_m_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_22_lstm_cell_22_bias_m_read_readvariableop1savev2_adam_dense_66_kernel_v_read_readvariableop/savev2_adam_dense_66_bias_v_read_readvariableop1savev2_adam_dense_67_kernel_v_read_readvariableop/savev2_adam_dense_67_bias_v_read_readvariableop1savev2_adam_dense_68_kernel_v_read_readvariableop/savev2_adam_dense_68_bias_v_read_readvariableop=savev2_adam_lstm_22_lstm_cell_22_kernel_v_read_readvariableopGsavev2_adam_lstm_22_lstm_cell_22_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_22_lstm_cell_22_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	љ
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:І
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ѕ
_input_shapesэ
З: ::::::: : : : : :::: : : : ::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$  

_output_shapes

:: !

_output_shapes
::$" 

_output_shapes

::$# 

_output_shapes

:: $

_output_shapes
::%

_output_shapes
: 
ЂЉ
╔
"__inference__traced_restore_444778
file_prefix2
 assignvariableop_dense_66_kernel:.
 assignvariableop_1_dense_66_bias:4
"assignvariableop_2_dense_67_kernel:.
 assignvariableop_3_dense_67_bias:4
"assignvariableop_4_dense_68_kernel:.
 assignvariableop_5_dense_68_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: A
/assignvariableop_11_lstm_22_lstm_cell_22_kernel:K
9assignvariableop_12_lstm_22_lstm_cell_22_recurrent_kernel:;
-assignvariableop_13_lstm_22_lstm_cell_22_bias:#
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: <
*assignvariableop_18_adam_dense_66_kernel_m:6
(assignvariableop_19_adam_dense_66_bias_m:<
*assignvariableop_20_adam_dense_67_kernel_m:6
(assignvariableop_21_adam_dense_67_bias_m:<
*assignvariableop_22_adam_dense_68_kernel_m:6
(assignvariableop_23_adam_dense_68_bias_m:H
6assignvariableop_24_adam_lstm_22_lstm_cell_22_kernel_m:R
@assignvariableop_25_adam_lstm_22_lstm_cell_22_recurrent_kernel_m:B
4assignvariableop_26_adam_lstm_22_lstm_cell_22_bias_m:<
*assignvariableop_27_adam_dense_66_kernel_v:6
(assignvariableop_28_adam_dense_66_bias_v:<
*assignvariableop_29_adam_dense_67_kernel_v:6
(assignvariableop_30_adam_dense_67_bias_v:<
*assignvariableop_31_adam_dense_68_kernel_v:6
(assignvariableop_32_adam_dense_68_bias_v:H
6assignvariableop_33_adam_lstm_22_lstm_cell_22_kernel_v:R
@assignvariableop_34_adam_lstm_22_lstm_cell_22_recurrent_kernel_v:B
4assignvariableop_35_adam_lstm_22_lstm_cell_22_bias_v:
identity_37ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9ѕ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*«
valueцBА%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH║
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ┌
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ф
_output_shapesЌ
ћ:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:І
AssignVariableOpAssignVariableOp assignvariableop_dense_66_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_66_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_67_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_67_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Љ
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_68_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ј
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_68_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:І
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ї
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ќ
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:а
AssignVariableOp_11AssignVariableOp/assignvariableop_11_lstm_22_lstm_cell_22_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ф
AssignVariableOp_12AssignVariableOp9assignvariableop_12_lstm_22_lstm_cell_22_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:ъ
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_22_lstm_cell_22_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:і
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:ї
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_66_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_66_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_67_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_67_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_68_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_68_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_22_lstm_cell_22_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_22_lstm_cell_22_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_22_lstm_cell_22_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_66_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_66_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_67_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_67_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Џ
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_68_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ў
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_68_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_lstm_22_lstm_cell_22_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:▒
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_lstm_22_lstm_cell_22_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ц
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_22_lstm_cell_22_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 у
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: н
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_37Identity_37:output:0*]
_input_shapesL
J: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
З
▓
(__inference_lstm_22_layer_call_fn_442990

inputs
unknown:
	unknown_0:
	unknown_1:
identityѕбStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *L
fGRE
C__inference_lstm_22_layer_call_and_return_conditional_losses_441519o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:         : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
┤╩
Т
C__inference_lstm_22_layer_call_and_return_conditional_losses_443615
inputs_0<
*lstm_cell_22_split_readvariableop_resource::
,lstm_cell_22_split_1_readvariableop_resource:6
$lstm_cell_22_readvariableop_resource:
identityѕбlstm_cell_22/ReadVariableOpбlstm_cell_22/ReadVariableOp_1бlstm_cell_22/ReadVariableOp_2бlstm_cell_22/ReadVariableOp_3б!lstm_cell_22/split/ReadVariableOpб#lstm_cell_22/split_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         R
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         c
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
         ┤
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмє
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       Я
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУм_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maskd
lstm_cell_22/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_22/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ъ
lstm_cell_22/ones_likeFill%lstm_cell_22/ones_like/Shape:output:0%lstm_cell_22/ones_like/Const:output:0*
T0*'
_output_shapes
:         _
lstm_cell_22/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ќ
lstm_cell_22/dropout/MulMullstm_cell_22/ones_like:output:0#lstm_cell_22/dropout/Const:output:0*
T0*'
_output_shapes
:         i
lstm_cell_22/dropout/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:д
1lstm_cell_22/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_22/dropout/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0h
#lstm_cell_22/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=═
!lstm_cell_22/dropout/GreaterEqualGreaterEqual:lstm_cell_22/dropout/random_uniform/RandomUniform:output:0,lstm_cell_22/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ѕ
lstm_cell_22/dropout/CastCast%lstm_cell_22/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         љ
lstm_cell_22/dropout/Mul_1Mullstm_cell_22/dropout/Mul:z:0lstm_cell_22/dropout/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_1/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_1/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_1/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_1/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_1/GreaterEqualGreaterEqual<lstm_cell_22/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_1/CastCast'lstm_cell_22/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_1/Mul_1Mullstm_cell_22/dropout_1/Mul:z:0lstm_cell_22/dropout_1/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_2/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_2/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_2/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_2/GreaterEqualGreaterEqual<lstm_cell_22/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_2/CastCast'lstm_cell_22/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_2/Mul_1Mullstm_cell_22/dropout_2/Mul:z:0lstm_cell_22/dropout_2/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Џ
lstm_cell_22/dropout_3/MulMullstm_cell_22/ones_like:output:0%lstm_cell_22/dropout_3/Const:output:0*
T0*'
_output_shapes
:         k
lstm_cell_22/dropout_3/ShapeShapelstm_cell_22/ones_like:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_3/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_3/GreaterEqualGreaterEqual<lstm_cell_22/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_3/CastCast'lstm_cell_22/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_3/Mul_1Mullstm_cell_22/dropout_3/Mul:z:0lstm_cell_22/dropout_3/Cast:y:0*
T0*'
_output_shapes
:         \
lstm_cell_22/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_22/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ђ?ц
lstm_cell_22/ones_like_1Fill'lstm_cell_22/ones_like_1/Shape:output:0'lstm_cell_22/ones_like_1/Const:output:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_4/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_4/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_4/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_4/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_4/GreaterEqualGreaterEqual<lstm_cell_22/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_4/CastCast'lstm_cell_22/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_4/Mul_1Mullstm_cell_22/dropout_4/Mul:z:0lstm_cell_22/dropout_4/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_5/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_5/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_5/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_5/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_5/GreaterEqualGreaterEqual<lstm_cell_22/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_5/CastCast'lstm_cell_22/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_5/Mul_1Mullstm_cell_22/dropout_5/Mul:z:0lstm_cell_22/dropout_5/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_6/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_6/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_6/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_6/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_6/GreaterEqualGreaterEqual<lstm_cell_22/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_6/CastCast'lstm_cell_22/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_6/Mul_1Mullstm_cell_22/dropout_6/Mul:z:0lstm_cell_22/dropout_6/Cast:y:0*
T0*'
_output_shapes
:         a
lstm_cell_22/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *С8ј?Ю
lstm_cell_22/dropout_7/MulMul!lstm_cell_22/ones_like_1:output:0%lstm_cell_22/dropout_7/Const:output:0*
T0*'
_output_shapes
:         m
lstm_cell_22/dropout_7/ShapeShape!lstm_cell_22/ones_like_1:output:0*
T0*
_output_shapes
:ф
3lstm_cell_22/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_22/dropout_7/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0j
%lstm_cell_22/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠╠=М
#lstm_cell_22/dropout_7/GreaterEqualGreaterEqual<lstm_cell_22/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_22/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:         Ї
lstm_cell_22/dropout_7/CastCast'lstm_cell_22/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:         ќ
lstm_cell_22/dropout_7/Mul_1Mullstm_cell_22/dropout_7/Mul:z:0lstm_cell_22/dropout_7/Cast:y:0*
T0*'
_output_shapes
:         Ѓ
lstm_cell_22/mulMulstrided_slice_2:output:0lstm_cell_22/dropout/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_1Mulstrided_slice_2:output:0 lstm_cell_22/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_2Mulstrided_slice_2:output:0 lstm_cell_22/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:         Є
lstm_cell_22/mul_3Mulstrided_slice_2:output:0 lstm_cell_22/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:         ^
lstm_cell_22/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ї
!lstm_cell_22/split/ReadVariableOpReadVariableOp*lstm_cell_22_split_readvariableop_resource*
_output_shapes

:*
dtype0┼
lstm_cell_22/splitSplit%lstm_cell_22/split/split_dim:output:0)lstm_cell_22/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitѓ
lstm_cell_22/MatMulMatMullstm_cell_22/mul:z:0lstm_cell_22/split:output:0*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_1MatMullstm_cell_22/mul_1:z:0lstm_cell_22/split:output:1*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_2MatMullstm_cell_22/mul_2:z:0lstm_cell_22/split:output:2*
T0*'
_output_shapes
:         є
lstm_cell_22/MatMul_3MatMullstm_cell_22/mul_3:z:0lstm_cell_22/split:output:3*
T0*'
_output_shapes
:         `
lstm_cell_22/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ї
#lstm_cell_22/split_1/ReadVariableOpReadVariableOp,lstm_cell_22_split_1_readvariableop_resource*
_output_shapes
:*
dtype0╗
lstm_cell_22/split_1Split'lstm_cell_22/split_1/split_dim:output:0+lstm_cell_22/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitЈ
lstm_cell_22/BiasAddBiasAddlstm_cell_22/MatMul:product:0lstm_cell_22/split_1:output:0*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_1BiasAddlstm_cell_22/MatMul_1:product:0lstm_cell_22/split_1:output:1*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_2BiasAddlstm_cell_22/MatMul_2:product:0lstm_cell_22/split_1:output:2*
T0*'
_output_shapes
:         Њ
lstm_cell_22/BiasAdd_3BiasAddlstm_cell_22/MatMul_3:product:0lstm_cell_22/split_1:output:3*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_4Mulzeros:output:0 lstm_cell_22/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_5Mulzeros:output:0 lstm_cell_22/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_6Mulzeros:output:0 lstm_cell_22/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:         }
lstm_cell_22/mul_7Mulzeros:output:0 lstm_cell_22/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:         ђ
lstm_cell_22/ReadVariableOpReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_22/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_22/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_22/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      г
lstm_cell_22/strided_sliceStridedSlice#lstm_cell_22/ReadVariableOp:value:0)lstm_cell_22/strided_slice/stack:output:0+lstm_cell_22/strided_slice/stack_1:output:0+lstm_cell_22/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskј
lstm_cell_22/MatMul_4MatMullstm_cell_22/mul_4:z:0#lstm_cell_22/strided_slice:output:0*
T0*'
_output_shapes
:         І
lstm_cell_22/addAddV2lstm_cell_22/BiasAdd:output:0lstm_cell_22/MatMul_4:product:0*
T0*'
_output_shapes
:         g
lstm_cell_22/SigmoidSigmoidlstm_cell_22/add:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_1ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_1StridedSlice%lstm_cell_22/ReadVariableOp_1:value:0+lstm_cell_22/strided_slice_1/stack:output:0-lstm_cell_22/strided_slice_1/stack_1:output:0-lstm_cell_22/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_5MatMullstm_cell_22/mul_5:z:0%lstm_cell_22/strided_slice_1:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_1AddV2lstm_cell_22/BiasAdd_1:output:0lstm_cell_22/MatMul_5:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_1Sigmoidlstm_cell_22/add_1:z:0*
T0*'
_output_shapes
:         y
lstm_cell_22/mul_8Mullstm_cell_22/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_2ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_2StridedSlice%lstm_cell_22/ReadVariableOp_2:value:0+lstm_cell_22/strided_slice_2/stack:output:0-lstm_cell_22/strided_slice_2/stack_1:output:0-lstm_cell_22/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_6MatMullstm_cell_22/mul_6:z:0%lstm_cell_22/strided_slice_2:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_2AddV2lstm_cell_22/BiasAdd_2:output:0lstm_cell_22/MatMul_6:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_2Sigmoidlstm_cell_22/add_2:z:0*
T0*'
_output_shapes
:         Ђ
lstm_cell_22/mul_9Mullstm_cell_22/Sigmoid:y:0lstm_cell_22/Sigmoid_2:y:0*
T0*'
_output_shapes
:         }
lstm_cell_22/add_3AddV2lstm_cell_22/mul_8:z:0lstm_cell_22/mul_9:z:0*
T0*'
_output_shapes
:         ѓ
lstm_cell_22/ReadVariableOp_3ReadVariableOp$lstm_cell_22_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_22/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_22/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_22/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Х
lstm_cell_22/strided_slice_3StridedSlice%lstm_cell_22/ReadVariableOp_3:value:0+lstm_cell_22/strided_slice_3/stack:output:0-lstm_cell_22/strided_slice_3/stack_1:output:0-lstm_cell_22/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskљ
lstm_cell_22/MatMul_7MatMullstm_cell_22/mul_7:z:0%lstm_cell_22/strided_slice_3:output:0*
T0*'
_output_shapes
:         Ј
lstm_cell_22/add_4AddV2lstm_cell_22/BiasAdd_3:output:0lstm_cell_22/MatMul_7:product:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_3Sigmoidlstm_cell_22/add_4:z:0*
T0*'
_output_shapes
:         k
lstm_cell_22/Sigmoid_4Sigmoidlstm_cell_22/add_3:z:0*
T0*'
_output_shapes
:         ё
lstm_cell_22/mul_10Mullstm_cell_22/Sigmoid_3:y:0lstm_cell_22/Sigmoid_4:y:0*
T0*'
_output_shapes
:         n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       И
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:жУмF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
         T
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : Э
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_22_split_readvariableop_resource,lstm_cell_22_split_1_readvariableop_resource$lstm_cell_22_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :         :         : : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_443417*
condR
while_cond_443416*K
output_shapes:
8: : : : :         :         : : : : : *
parallel_iterations Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"       ╦
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :                  *
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
         a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:         *
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  [
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:         ќ
NoOpNoOp^lstm_cell_22/ReadVariableOp^lstm_cell_22/ReadVariableOp_1^lstm_cell_22/ReadVariableOp_2^lstm_cell_22/ReadVariableOp_3"^lstm_cell_22/split/ReadVariableOp$^lstm_cell_22/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:                  : : : 2:
lstm_cell_22/ReadVariableOplstm_cell_22/ReadVariableOp2>
lstm_cell_22/ReadVariableOp_1lstm_cell_22/ReadVariableOp_12>
lstm_cell_22/ReadVariableOp_2lstm_cell_22/ReadVariableOp_22>
lstm_cell_22/ReadVariableOp_3lstm_cell_22/ReadVariableOp_32F
!lstm_cell_22/split/ReadVariableOp!lstm_cell_22/split/ReadVariableOp2J
#lstm_cell_22/split_1/ReadVariableOp#lstm_cell_22/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs/0
┬
ќ
)__inference_dense_67_layer_call_fn_444238

inputs
unknown:
	unknown_0:
identityѕбStatefulPartitionedCall┘
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8ѓ *M
fHRF
D__inference_dense_67_layer_call_and_return_conditional_losses_441537o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs"█L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*й
serving_defaultЕ
M
dense_66_input;
 serving_default_dense_66_input:0         <
dense_680
StatefulPartitionedCall:0         tensorflow/serving/predict:Єё
ѓ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
╗

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
┌
cell

state_spec
	variables
trainable_variables
regularization_losses
	keras_api
_random_generator
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_rnn_layer
╗

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
╗

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
ш
/iter

0beta_1

1beta_2
	2decay
3learning_ratemjmkml mm'mn(mo4mp5mq6mrvsvtvu vv'vw(vx4vy5vz6v{"
	optimizer
_
0
1
42
53
64
5
 6
'7
(8"
trackable_list_wrapper
_
0
1
42
53
64
5
 6
'7
(8"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
є2Ѓ
.__inference_sequential_21_layer_call_fn_441581
.__inference_sequential_21_layer_call_fn_442180
.__inference_sequential_21_layer_call_fn_442203
.__inference_sequential_21_layer_call_fn_442099└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Ы2№
I__inference_sequential_21_layer_call_and_return_conditional_losses_442484
I__inference_sequential_21_layer_call_and_return_conditional_losses_442893
I__inference_sequential_21_layer_call_and_return_conditional_losses_442125
I__inference_sequential_21_layer_call_and_return_conditional_losses_442151└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
МBл
!__inference__wrapped_model_440720dense_66_input"ў
Љ▓Ї
FullArgSpec
argsџ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
,
<serving_default"
signature_map
!:2dense_66/kernel
:2dense_66/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
=non_trainable_variables

>layers
?metrics
@layer_regularization_losses
Alayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
М2л
)__inference_dense_66_layer_call_fn_442927б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_66_layer_call_and_return_conditional_losses_442957б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Э
B
state_size

4kernel
5recurrent_kernel
6bias
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G_random_generator
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
╣

Jstates
Knon_trainable_variables

Llayers
Mmetrics
Nlayer_regularization_losses
Olayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
Ѓ2ђ
(__inference_lstm_22_layer_call_fn_442968
(__inference_lstm_22_layer_call_fn_442979
(__inference_lstm_22_layer_call_fn_442990
(__inference_lstm_22_layer_call_fn_443001Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
№2В
C__inference_lstm_22_layer_call_and_return_conditional_losses_443244
C__inference_lstm_22_layer_call_and_return_conditional_losses_443615
C__inference_lstm_22_layer_call_and_return_conditional_losses_443858
C__inference_lstm_22_layer_call_and_return_conditional_losses_444229Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
!:2dense_67/kernel
:2dense_67/bias
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Pnon_trainable_variables

Qlayers
Rmetrics
Slayer_regularization_losses
Tlayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
М2л
)__inference_dense_67_layer_call_fn_444238б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_67_layer_call_and_return_conditional_losses_444248б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
!:2dense_68/kernel
:2dense_68/bias
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
Unon_trainable_variables

Vlayers
Wmetrics
Xlayer_regularization_losses
Ylayer_metrics
)	variables
*trainable_variables
+regularization_losses
-__call__
*.&call_and_return_all_conditional_losses
&."call_and_return_conditional_losses"
_generic_user_object
М2л
)__inference_dense_68_layer_call_fn_444257б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ь2в
D__inference_dense_68_layer_call_and_return_conditional_losses_444267б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+2lstm_22/lstm_cell_22/kernel
7:52%lstm_22/lstm_cell_22/recurrent_kernel
':%2lstm_22/lstm_cell_22/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
мB¤
$__inference_signature_wrapper_442918dense_66_input"ћ
Ї▓Ѕ
FullArgSpec
argsџ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
5
40
51
62"
trackable_list_wrapper
 "
trackable_list_wrapper
Г
\non_trainable_variables

]layers
^metrics
_layer_regularization_losses
`layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
"
_generic_user_object
б2Ъ
-__inference_lstm_cell_22_layer_call_fn_444284
-__inference_lstm_cell_22_layer_call_fn_444301Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
п2Н
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_444383
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_444529Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	atotal
	bcount
c	variables
d	keras_api"
_tf_keras_metric
^
	etotal
	fcount
g
_fn_kwargs
h	variables
i	keras_api"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
.
a0
b1"
trackable_list_wrapper
-
c	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
e0
f1"
trackable_list_wrapper
-
h	variables"
_generic_user_object
&:$2Adam/dense_66/kernel/m
 :2Adam/dense_66/bias/m
&:$2Adam/dense_67/kernel/m
 :2Adam/dense_67/bias/m
&:$2Adam/dense_68/kernel/m
 :2Adam/dense_68/bias/m
2:02"Adam/lstm_22/lstm_cell_22/kernel/m
<::2,Adam/lstm_22/lstm_cell_22/recurrent_kernel/m
,:*2 Adam/lstm_22/lstm_cell_22/bias/m
&:$2Adam/dense_66/kernel/v
 :2Adam/dense_66/bias/v
&:$2Adam/dense_67/kernel/v
 :2Adam/dense_67/bias/v
&:$2Adam/dense_68/kernel/v
 :2Adam/dense_68/bias/v
2:02"Adam/lstm_22/lstm_cell_22/kernel/v
<::2,Adam/lstm_22/lstm_cell_22/recurrent_kernel/v
,:*2 Adam/lstm_22/lstm_cell_22/bias/vб
!__inference__wrapped_model_440720}	465 '(;б8
1б.
,і)
dense_66_input         
ф "3ф0
.
dense_68"і
dense_68         г
D__inference_dense_66_layer_call_and_return_conditional_losses_442957d3б0
)б&
$і!
inputs         
ф ")б&
і
0         
џ ё
)__inference_dense_66_layer_call_fn_442927W3б0
)б&
$і!
inputs         
ф "і         ц
D__inference_dense_67_layer_call_and_return_conditional_losses_444248\ /б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
)__inference_dense_67_layer_call_fn_444238O /б,
%б"
 і
inputs         
ф "і         ц
D__inference_dense_68_layer_call_and_return_conditional_losses_444267\'(/б,
%б"
 і
inputs         
ф "%б"
і
0         
џ |
)__inference_dense_68_layer_call_fn_444257O'(/б,
%б"
 і
inputs         
ф "і         ─
C__inference_lstm_22_layer_call_and_return_conditional_losses_443244}465OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "%б"
і
0         
џ ─
C__inference_lstm_22_layer_call_and_return_conditional_losses_443615}465OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "%б"
і
0         
џ ┤
C__inference_lstm_22_layer_call_and_return_conditional_losses_443858m465?б<
5б2
$і!
inputs         

 
p 

 
ф "%б"
і
0         
џ ┤
C__inference_lstm_22_layer_call_and_return_conditional_losses_444229m465?б<
5б2
$і!
inputs         

 
p

 
ф "%б"
і
0         
џ ю
(__inference_lstm_22_layer_call_fn_442968p465OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "і         ю
(__inference_lstm_22_layer_call_fn_442979p465OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "і         ї
(__inference_lstm_22_layer_call_fn_442990`465?б<
5б2
$і!
inputs         

 
p 

 
ф "і         ї
(__inference_lstm_22_layer_call_fn_443001`465?б<
5б2
$і!
inputs         

 
p

 
ф "і         ╩
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_444383§465ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ ╩
H__inference_lstm_cell_22_layer_call_and_return_conditional_losses_444529§465ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "sбp
iбf
і
0/0         
EџB
і
0/1/0         
і
0/1/1         
џ Ъ
-__inference_lstm_cell_22_layer_call_fn_444284ь465ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p 
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         Ъ
-__inference_lstm_cell_22_layer_call_fn_444301ь465ђб}
vбs
 і
inputs         
KбH
"і
states/0         
"і
states/1         
p
ф "cб`
і
0         
Aџ>
і
1/0         
і
1/1         ─
I__inference_sequential_21_layer_call_and_return_conditional_losses_442125w	465 '(Cб@
9б6
,і)
dense_66_input         
p 

 
ф "%б"
і
0         
џ ─
I__inference_sequential_21_layer_call_and_return_conditional_losses_442151w	465 '(Cб@
9б6
,і)
dense_66_input         
p

 
ф "%б"
і
0         
џ ╝
I__inference_sequential_21_layer_call_and_return_conditional_losses_442484o	465 '(;б8
1б.
$і!
inputs         
p 

 
ф "%б"
і
0         
џ ╝
I__inference_sequential_21_layer_call_and_return_conditional_losses_442893o	465 '(;б8
1б.
$і!
inputs         
p

 
ф "%б"
і
0         
џ ю
.__inference_sequential_21_layer_call_fn_441581j	465 '(Cб@
9б6
,і)
dense_66_input         
p 

 
ф "і         ю
.__inference_sequential_21_layer_call_fn_442099j	465 '(Cб@
9б6
,і)
dense_66_input         
p

 
ф "і         ћ
.__inference_sequential_21_layer_call_fn_442180b	465 '(;б8
1б.
$і!
inputs         
p 

 
ф "і         ћ
.__inference_sequential_21_layer_call_fn_442203b	465 '(;б8
1б.
$і!
inputs         
p

 
ф "і         И
$__inference_signature_wrapper_442918Ј	465 '(MбJ
б 
Cф@
>
dense_66_input,і)
dense_66_input         "3ф0
.
dense_68"і
dense_68         