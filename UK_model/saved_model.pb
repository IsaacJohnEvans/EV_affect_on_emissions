â$
ì
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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
­
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
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
ö
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
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

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
"serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68¿#
z
dense_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_72/kernel
s
#dense_72/kernel/Read/ReadVariableOpReadVariableOpdense_72/kernel*
_output_shapes

:*
dtype0
r
dense_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_72/bias
k
!dense_72/bias/Read/ReadVariableOpReadVariableOpdense_72/bias*
_output_shapes
:*
dtype0
z
dense_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_73/kernel
s
#dense_73/kernel/Read/ReadVariableOpReadVariableOpdense_73/kernel*
_output_shapes

:*
dtype0
r
dense_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_73/bias
k
!dense_73/bias/Read/ReadVariableOpReadVariableOpdense_73/bias*
_output_shapes
:*
dtype0
z
dense_74/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:* 
shared_namedense_74/kernel
s
#dense_74/kernel/Read/ReadVariableOpReadVariableOpdense_74/kernel*
_output_shapes

:*
dtype0
r
dense_74/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_74/bias
k
!dense_74/bias/Read/ReadVariableOpReadVariableOpdense_74/bias*
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

lstm_24/lstm_cell_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namelstm_24/lstm_cell_24/kernel

/lstm_24/lstm_cell_24/kernel/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_24/kernel*
_output_shapes

:*
dtype0
¦
%lstm_24/lstm_cell_24/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*6
shared_name'%lstm_24/lstm_cell_24/recurrent_kernel

9lstm_24/lstm_cell_24/recurrent_kernel/Read/ReadVariableOpReadVariableOp%lstm_24/lstm_cell_24/recurrent_kernel*
_output_shapes

:*
dtype0

lstm_24/lstm_cell_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namelstm_24/lstm_cell_24/bias

-lstm_24/lstm_cell_24/bias/Read/ReadVariableOpReadVariableOplstm_24/lstm_cell_24/bias*
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

Adam/dense_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_72/kernel/m

*Adam/dense_72/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_72/bias/m
y
(Adam/dense_72/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/m*
_output_shapes
:*
dtype0

Adam/dense_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_73/kernel/m

*Adam/dense_73/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_73/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_73/bias/m
y
(Adam/dense_73/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_73/bias/m*
_output_shapes
:*
dtype0

Adam/dense_74/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_74/kernel/m

*Adam/dense_74/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_74/kernel/m*
_output_shapes

:*
dtype0

Adam/dense_74/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_74/bias/m
y
(Adam/dense_74/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_74/bias/m*
_output_shapes
:*
dtype0
 
"Adam/lstm_24/lstm_cell_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/lstm_24/lstm_cell_24/kernel/m

6Adam/lstm_24/lstm_cell_24/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_24/kernel/m*
_output_shapes

:*
dtype0
´
,Adam/lstm_24/lstm_cell_24/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m
­
@Adam/lstm_24/lstm_cell_24/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m*
_output_shapes

:*
dtype0

 Adam/lstm_24/lstm_cell_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_24/lstm_cell_24/bias/m

4Adam/lstm_24/lstm_cell_24/bias/m/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_24/bias/m*
_output_shapes
:*
dtype0

Adam/dense_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_72/kernel/v

*Adam/dense_72/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_72/bias/v
y
(Adam/dense_72/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_72/bias/v*
_output_shapes
:*
dtype0

Adam/dense_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_73/kernel/v

*Adam/dense_73/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_73/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_73/bias/v
y
(Adam/dense_73/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_73/bias/v*
_output_shapes
:*
dtype0

Adam/dense_74/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameAdam/dense_74/kernel/v

*Adam/dense_74/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_74/kernel/v*
_output_shapes

:*
dtype0

Adam/dense_74/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_74/bias/v
y
(Adam/dense_74/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_74/bias/v*
_output_shapes
:*
dtype0
 
"Adam/lstm_24/lstm_cell_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*3
shared_name$"Adam/lstm_24/lstm_cell_24/kernel/v

6Adam/lstm_24/lstm_cell_24/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/lstm_24/lstm_cell_24/kernel/v*
_output_shapes

:*
dtype0
´
,Adam/lstm_24/lstm_cell_24/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*=
shared_name.,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v
­
@Adam/lstm_24/lstm_cell_24/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v*
_output_shapes

:*
dtype0

 Adam/lstm_24/lstm_cell_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/lstm_24/lstm_cell_24/bias/v

4Adam/lstm_24/lstm_cell_24/bias/v/Read/ReadVariableOpReadVariableOp Adam/lstm_24/lstm_cell_24/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
ø?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*³?
value©?B¦? B?
è
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
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
Á
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
¦

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses*
¦

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses*
æ
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
°
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
VARIABLE_VALUEdense_72/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_72/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

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
ã
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


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
VARIABLE_VALUEdense_73/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_73/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
 1*

0
 1*
* 

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
VARIABLE_VALUEdense_74/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_74/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

'0
(1*

'0
(1*
* 

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
VARIABLE_VALUElstm_24/lstm_cell_24/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUE%lstm_24/lstm_cell_24/recurrent_kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_24/lstm_cell_24/bias&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
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

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
|
VARIABLE_VALUEAdam/dense_72/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_72/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_73/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_73/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_74/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_74/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_24/lstm_cell_24/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_24/lstm_cell_24/recurrent_kernel/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_24/lstm_cell_24/bias/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_72/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_72/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_73/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_73/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_74/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_74/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUE"Adam/lstm_24/lstm_cell_24/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE,Adam/lstm_24/lstm_cell_24/recurrent_kernel/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_24/lstm_cell_24/bias/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_dense_72_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_dense_72_inputdense_72/kerneldense_72/biaslstm_24/lstm_cell_24/kernellstm_24/lstm_cell_24/bias%lstm_24/lstm_cell_24/recurrent_kerneldense_73/kerneldense_73/biasdense_74/kerneldense_74/bias*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_483371
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¯
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#dense_72/kernel/Read/ReadVariableOp!dense_72/bias/Read/ReadVariableOp#dense_73/kernel/Read/ReadVariableOp!dense_73/bias/Read/ReadVariableOp#dense_74/kernel/Read/ReadVariableOp!dense_74/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/lstm_24/lstm_cell_24/kernel/Read/ReadVariableOp9lstm_24/lstm_cell_24/recurrent_kernel/Read/ReadVariableOp-lstm_24/lstm_cell_24/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp*Adam/dense_72/kernel/m/Read/ReadVariableOp(Adam/dense_72/bias/m/Read/ReadVariableOp*Adam/dense_73/kernel/m/Read/ReadVariableOp(Adam/dense_73/bias/m/Read/ReadVariableOp*Adam/dense_74/kernel/m/Read/ReadVariableOp(Adam/dense_74/bias/m/Read/ReadVariableOp6Adam/lstm_24/lstm_cell_24/kernel/m/Read/ReadVariableOp@Adam/lstm_24/lstm_cell_24/recurrent_kernel/m/Read/ReadVariableOp4Adam/lstm_24/lstm_cell_24/bias/m/Read/ReadVariableOp*Adam/dense_72/kernel/v/Read/ReadVariableOp(Adam/dense_72/bias/v/Read/ReadVariableOp*Adam/dense_73/kernel/v/Read/ReadVariableOp(Adam/dense_73/bias/v/Read/ReadVariableOp*Adam/dense_74/kernel/v/Read/ReadVariableOp(Adam/dense_74/bias/v/Read/ReadVariableOp6Adam/lstm_24/lstm_cell_24/kernel/v/Read/ReadVariableOp@Adam/lstm_24/lstm_cell_24/recurrent_kernel/v/Read/ReadVariableOp4Adam/lstm_24/lstm_cell_24/bias/v/Read/ReadVariableOpConst*1
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
GPU 2J 8 *(
f#R!
__inference__traced_save_485113
Ú
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_72/kerneldense_72/biasdense_73/kerneldense_73/biasdense_74/kerneldense_74/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratelstm_24/lstm_cell_24/kernel%lstm_24/lstm_cell_24/recurrent_kernellstm_24/lstm_cell_24/biastotalcounttotal_1count_1Adam/dense_72/kernel/mAdam/dense_72/bias/mAdam/dense_73/kernel/mAdam/dense_73/bias/mAdam/dense_74/kernel/mAdam/dense_74/bias/m"Adam/lstm_24/lstm_cell_24/kernel/m,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m Adam/lstm_24/lstm_cell_24/bias/mAdam/dense_72/kernel/vAdam/dense_72/bias/vAdam/dense_73/kernel/vAdam/dense_73/bias/vAdam/dense_74/kernel/vAdam/dense_74/bias/v"Adam/lstm_24/lstm_cell_24/kernel/v,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v Adam/lstm_24/lstm_cell_24/bias/v*0
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_485231ë"
ß
ä
C__inference_lstm_24_layer_call_and_return_conditional_losses_481972

inputs<
*lstm_cell_24_split_readvariableop_resource::
,lstm_cell_24_split_1_readvariableop_resource:6
$lstm_cell_24_readvariableop_resource:
identity¢lstm_cell_24/ReadVariableOp¢lstm_cell_24/ReadVariableOp_1¢lstm_cell_24/ReadVariableOp_2¢lstm_cell_24/ReadVariableOp_3¢!lstm_cell_24/split/ReadVariableOp¢#lstm_cell_24/split_1/ReadVariableOp¢while;
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskd
lstm_cell_24/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_24/ones_likeFill%lstm_cell_24/ones_like/Shape:output:0%lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
lstm_cell_24/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm_cell_24/ones_like_1Fill'lstm_cell_24/ones_like_1/Shape:output:0'lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mulMulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_1Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_2Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_3Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_24/split/ReadVariableOpReadVariableOp*lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Å
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0)lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_cell_24/MatMulMatMullstm_cell_24/mul:z:0lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_1MatMullstm_cell_24/mul_1:z:0lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_2MatMullstm_cell_24/mul_2:z:0lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_3MatMullstm_cell_24/mul_3:z:0lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_24/split_1/ReadVariableOpReadVariableOp,lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0»
lstm_cell_24/split_1Split'lstm_cell_24/split_1/split_dim:output:0+lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
lstm_cell_24/BiasAddBiasAddlstm_cell_24/MatMul:product:0lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_1BiasAddlstm_cell_24/MatMul_1:product:0lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_2BiasAddlstm_cell_24/MatMul_2:product:0lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_3BiasAddlstm_cell_24/MatMul_3:product:0lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_4Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_5Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_6Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_7Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOpReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¬
lstm_cell_24/strided_sliceStridedSlice#lstm_cell_24/ReadVariableOp:value:0)lstm_cell_24/strided_slice/stack:output:0+lstm_cell_24/strided_slice/stack_1:output:0+lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_4MatMullstm_cell_24/mul_4:z:0#lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/addAddV2lstm_cell_24/BiasAdd:output:0lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_24/SigmoidSigmoidlstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_1ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_1StridedSlice%lstm_cell_24/ReadVariableOp_1:value:0+lstm_cell_24/strided_slice_1/stack:output:0-lstm_cell_24/strided_slice_1/stack_1:output:0-lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_5MatMullstm_cell_24/mul_5:z:0%lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_1AddV2lstm_cell_24/BiasAdd_1:output:0lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_24/mul_8Mullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_2ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_2StridedSlice%lstm_cell_24/ReadVariableOp_2:value:0+lstm_cell_24/strided_slice_2/stack:output:0-lstm_cell_24/strided_slice_2/stack_1:output:0-lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_6MatMullstm_cell_24/mul_6:z:0%lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_2AddV2lstm_cell_24/BiasAdd_2:output:0lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_9Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/add_3AddV2lstm_cell_24/mul_8:z:0lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_3ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_3StridedSlice%lstm_cell_24/ReadVariableOp_3:value:0+lstm_cell_24/strided_slice_3/stack:output:0-lstm_cell_24/strided_slice_3/stack_1:output:0-lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_7MatMullstm_cell_24/mul_7:z:0%lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_4AddV2lstm_cell_24/BiasAdd_3:output:0lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_3Sigmoidlstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_4Sigmoidlstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_10Mullstm_cell_24/Sigmoid_3:y:0lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_24_split_readvariableop_resource,lstm_cell_24_split_1_readvariableop_resource$lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_481838*
condR
while_cond_481837*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell_24/ReadVariableOp^lstm_cell_24/ReadVariableOp_1^lstm_cell_24/ReadVariableOp_2^lstm_cell_24/ReadVariableOp_3"^lstm_cell_24/split/ReadVariableOp$^lstm_cell_24/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2:
lstm_cell_24/ReadVariableOplstm_cell_24/ReadVariableOp2>
lstm_cell_24/ReadVariableOp_1lstm_cell_24/ReadVariableOp_12>
lstm_cell_24/ReadVariableOp_2lstm_cell_24/ReadVariableOp_22>
lstm_cell_24/ReadVariableOp_3lstm_cell_24/ReadVariableOp_32F
!lstm_cell_24/split/ReadVariableOp!lstm_cell_24/split/ReadVariableOp2J
#lstm_cell_24/split_1/ReadVariableOp#lstm_cell_24/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÔÄ
	
while_body_484484
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_24_split_readvariableop_resource_0:B
4while_lstm_cell_24_split_1_readvariableop_resource_0:>
,while_lstm_cell_24_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_24_split_readvariableop_resource:@
2while_lstm_cell_24_split_1_readvariableop_resource:<
*while_lstm_cell_24_readvariableop_resource:¢!while/lstm_cell_24/ReadVariableOp¢#while/lstm_cell_24/ReadVariableOp_1¢#while/lstm_cell_24/ReadVariableOp_2¢#while/lstm_cell_24/ReadVariableOp_3¢'while/lstm_cell_24/split/ReadVariableOp¢)while/lstm_cell_24/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
"while/lstm_cell_24/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
while/lstm_cell_24/ones_likeFill+while/lstm_cell_24/ones_like/Shape:output:0+while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 while/lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?©
while/lstm_cell_24/dropout/MulMul%while/lstm_cell_24/ones_like:output:0)while/lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 while/lstm_cell_24/dropout/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:²
7while/lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0n
)while/lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ß
'while/lstm_cell_24/dropout/GreaterEqualGreaterEqual@while/lstm_cell_24/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/dropout/CastCast+while/lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 while/lstm_cell_24/dropout/Mul_1Mul"while/lstm_cell_24/dropout/Mul:z:0#while/lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_1/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_1/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_1/CastCast-while/lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_1/Mul_1Mul$while/lstm_cell_24/dropout_1/Mul:z:0%while/lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_2/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_2/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_2/CastCast-while/lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_2/Mul_1Mul$while/lstm_cell_24/dropout_2/Mul:z:0%while/lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_3/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_3/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_3/CastCast-while/lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_3/Mul_1Mul$while/lstm_cell_24/dropout_3/Mul:z:0%while/lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$while/lstm_cell_24/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
while/lstm_cell_24/ones_like_1Fill-while/lstm_cell_24/ones_like_1/Shape:output:0-while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_4/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_4/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_4/CastCast-while/lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_4/Mul_1Mul$while/lstm_cell_24/dropout_4/Mul:z:0%while/lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_5/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_5/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_5/CastCast-while/lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_5/Mul_1Mul$while/lstm_cell_24/dropout_5/Mul:z:0%while/lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_6/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_6/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_6/CastCast-while/lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_6/Mul_1Mul$while/lstm_cell_24/dropout_6/Mul:z:0%while/lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_7/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_7/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_7/CastCast-while/lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_7/Mul_1Mul$while/lstm_cell_24/dropout_7/Mul:z:0%while/lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell_24/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_24/split/ReadVariableOpReadVariableOp2while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0×
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
while/lstm_cell_24/MatMulMatMulwhile/lstm_cell_24/mul:z:0!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_1MatMulwhile/lstm_cell_24/mul_1:z:0!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_2MatMulwhile/lstm_cell_24/mul_2:z:0!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_3MatMulwhile/lstm_cell_24/mul_3:z:0!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0Í
while/lstm_cell_24/split_1Split-while/lstm_cell_24/split_1/split_dim:output:01while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¡
while/lstm_cell_24/BiasAddBiasAdd#while/lstm_cell_24/MatMul:product:0#while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_1BiasAdd%while/lstm_cell_24/MatMul_1:product:0#while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_2BiasAdd%while/lstm_cell_24/MatMul_2:product:0#while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_3BiasAdd%while/lstm_cell_24/MatMul_3:product:0#while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_4Mulwhile_placeholder_2&while/lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_5Mulwhile_placeholder_2&while/lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_6Mulwhile_placeholder_2&while/lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_7Mulwhile_placeholder_2&while/lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/ReadVariableOpReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 while/lstm_cell_24/strided_sliceStridedSlice)while/lstm_cell_24/ReadVariableOp:value:0/while/lstm_cell_24/strided_slice/stack:output:01while/lstm_cell_24/strided_slice/stack_1:output:01while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask 
while/lstm_cell_24/MatMul_4MatMulwhile/lstm_cell_24/mul_4:z:0)while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/addAddV2#while/lstm_cell_24/BiasAdd:output:0%while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_24/SigmoidSigmoidwhile/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_1ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_1StridedSlice+while/lstm_cell_24/ReadVariableOp_1:value:01while/lstm_cell_24/strided_slice_1/stack:output:03while/lstm_cell_24/strided_slice_1/stack_1:output:03while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_5MatMulwhile/lstm_cell_24/mul_5:z:0+while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_1AddV2%while/lstm_cell_24/BiasAdd_1:output:0%while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_1Sigmoidwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_8Mul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_2ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_2StridedSlice+while/lstm_cell_24/ReadVariableOp_2:value:01while/lstm_cell_24/strided_slice_2/stack:output:03while/lstm_cell_24/strided_slice_2/stack_1:output:03while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_6MatMulwhile/lstm_cell_24/mul_6:z:0+while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_2AddV2%while/lstm_cell_24/BiasAdd_2:output:0%while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_2Sigmoidwhile/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_9Mulwhile/lstm_cell_24/Sigmoid:y:0 while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/add_3AddV2while/lstm_cell_24/mul_8:z:0while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_3ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_3StridedSlice+while/lstm_cell_24/ReadVariableOp_3:value:01while/lstm_cell_24/strided_slice_3/stack:output:03while/lstm_cell_24/strided_slice_3/stack_1:output:03while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_7MatMulwhile/lstm_cell_24/mul_7:z:0+while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_4AddV2%while/lstm_cell_24/BiasAdd_3:output:0%while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_3Sigmoidwhile/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_4Sigmoidwhile/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_10Mul while/lstm_cell_24/Sigmoid_3:y:0 while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_24/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_24/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸

while/NoOpNoOp"^while/lstm_cell_24/ReadVariableOp$^while/lstm_cell_24/ReadVariableOp_1$^while/lstm_cell_24/ReadVariableOp_2$^while/lstm_cell_24/ReadVariableOp_3(^while/lstm_cell_24/split/ReadVariableOp*^while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_24_readvariableop_resource,while_lstm_cell_24_readvariableop_resource_0"j
2while_lstm_cell_24_split_1_readvariableop_resource4while_lstm_cell_24_split_1_readvariableop_resource_0"f
0while_lstm_cell_24_split_readvariableop_resource2while_lstm_cell_24_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2F
!while/lstm_cell_24/ReadVariableOp!while/lstm_cell_24/ReadVariableOp2J
#while/lstm_cell_24/ReadVariableOp_1#while/lstm_cell_24/ReadVariableOp_12J
#while/lstm_cell_24/ReadVariableOp_2#while/lstm_cell_24/ReadVariableOp_22J
#while/lstm_cell_24/ReadVariableOp_3#while/lstm_cell_24/ReadVariableOp_32R
'while/lstm_cell_24/split/ReadVariableOp'while/lstm_cell_24/split/ReadVariableOp2V
)while/lstm_cell_24/split_1/ReadVariableOp)while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
À
â
I__inference_sequential_23_layer_call_and_return_conditional_losses_482937

inputs<
*dense_72_tensordot_readvariableop_resource:6
(dense_72_biasadd_readvariableop_resource:D
2lstm_24_lstm_cell_24_split_readvariableop_resource:B
4lstm_24_lstm_cell_24_split_1_readvariableop_resource:>
,lstm_24_lstm_cell_24_readvariableop_resource:9
'dense_73_matmul_readvariableop_resource:6
(dense_73_biasadd_readvariableop_resource:9
'dense_74_matmul_readvariableop_resource:6
(dense_74_biasadd_readvariableop_resource:
identity¢dense_72/BiasAdd/ReadVariableOp¢!dense_72/Tensordot/ReadVariableOp¢dense_73/BiasAdd/ReadVariableOp¢dense_73/MatMul/ReadVariableOp¢dense_74/BiasAdd/ReadVariableOp¢dense_74/MatMul/ReadVariableOp¢#lstm_24/lstm_cell_24/ReadVariableOp¢%lstm_24/lstm_cell_24/ReadVariableOp_1¢%lstm_24/lstm_cell_24/ReadVariableOp_2¢%lstm_24/lstm_cell_24/ReadVariableOp_3¢)lstm_24/lstm_cell_24/split/ReadVariableOp¢+lstm_24/lstm_cell_24/split_1/ReadVariableOp¢lstm_24/while
!dense_72/Tensordot/ReadVariableOpReadVariableOp*dense_72_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_72/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_72/Tensordot/GatherV2GatherV2!dense_72/Tensordot/Shape:output:0 dense_72/Tensordot/free:output:0)dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_72/Tensordot/GatherV2_1GatherV2!dense_72/Tensordot/Shape:output:0 dense_72/Tensordot/axes:output:0+dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_72/Tensordot/ProdProd$dense_72/Tensordot/GatherV2:output:0!dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_72/Tensordot/Prod_1Prod&dense_72/Tensordot/GatherV2_1:output:0#dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_72/Tensordot/concatConcatV2 dense_72/Tensordot/free:output:0 dense_72/Tensordot/axes:output:0'dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_72/Tensordot/stackPack dense_72/Tensordot/Prod:output:0"dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_72/Tensordot/transpose	Transposeinputs"dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
dense_72/Tensordot/ReshapeReshape dense_72/Tensordot/transpose:y:0!dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_72/Tensordot/MatMulMatMul#dense_72/Tensordot/Reshape:output:0)dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_72/Tensordot/concat_1ConcatV2$dense_72/Tensordot/GatherV2:output:0#dense_72/Tensordot/Const_2:output:0)dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_72/TensordotReshape#dense_72/Tensordot/MatMul:product:0$dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_72/BiasAddBiasAdddense_72/Tensordot:output:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
lstm_24/ShapeShapedense_72/BiasAdd:output:0*
T0*
_output_shapes
:e
lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_24/strided_sliceStridedSlicelstm_24/Shape:output:0$lstm_24/strided_slice/stack:output:0&lstm_24/strided_slice/stack_1:output:0&lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_24/zeros/packedPacklstm_24/strided_slice:output:0lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_24/zeros_1/packedPacklstm_24/strided_slice:output:0!lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_24/transpose	Transposedense_72/BiasAdd:output:0lstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_24/Shape_1Shapelstm_24/transpose:y:0*
T0*
_output_shapes
:g
lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_24/strided_slice_1StridedSlicelstm_24/Shape_1:output:0&lstm_24/strided_slice_1/stack:output:0(lstm_24/strided_slice_1/stack_1:output:0(lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_24/TensorArrayV2TensorListReserve,lstm_24/TensorArrayV2/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_24/transpose:y:0Flstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_24/strided_slice_2StridedSlicelstm_24/transpose:y:0&lstm_24/strided_slice_2/stack:output:0(lstm_24/strided_slice_2/stack_1:output:0(lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskt
$lstm_24/lstm_cell_24/ones_like/ShapeShape lstm_24/strided_slice_2:output:0*
T0*
_output_shapes
:i
$lstm_24/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
lstm_24/lstm_cell_24/ones_likeFill-lstm_24/lstm_cell_24/ones_like/Shape:output:0-lstm_24/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
&lstm_24/lstm_cell_24/ones_like_1/ShapeShapelstm_24/zeros:output:0*
T0*
_output_shapes
:k
&lstm_24/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
 lstm_24/lstm_cell_24/ones_like_1Fill/lstm_24/lstm_cell_24/ones_like_1/Shape:output:0/lstm_24/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mulMul lstm_24/strided_slice_2:output:0'lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_1Mul lstm_24/strided_slice_2:output:0'lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_2Mul lstm_24/strided_slice_2:output:0'lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_3Mul lstm_24/strided_slice_2:output:0'lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm_24/lstm_cell_24/split/ReadVariableOpReadVariableOp2lstm_24_lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Ý
lstm_24/lstm_cell_24/splitSplit-lstm_24/lstm_cell_24/split/split_dim:output:01lstm_24/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_24/lstm_cell_24/MatMulMatMullstm_24/lstm_cell_24/mul:z:0#lstm_24/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/MatMul_1MatMullstm_24/lstm_cell_24/mul_1:z:0#lstm_24/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/MatMul_2MatMullstm_24/lstm_cell_24/mul_2:z:0#lstm_24/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/MatMul_3MatMullstm_24/lstm_cell_24/mul_3:z:0#lstm_24/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&lstm_24/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm_24/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0Ó
lstm_24/lstm_cell_24/split_1Split/lstm_24/lstm_cell_24/split_1/split_dim:output:03lstm_24/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split§
lstm_24/lstm_cell_24/BiasAddBiasAdd%lstm_24/lstm_cell_24/MatMul:product:0%lstm_24/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_24/lstm_cell_24/BiasAdd_1BiasAdd'lstm_24/lstm_cell_24/MatMul_1:product:0%lstm_24/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_24/lstm_cell_24/BiasAdd_2BiasAdd'lstm_24/lstm_cell_24/MatMul_2:product:0%lstm_24/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_24/lstm_cell_24/BiasAdd_3BiasAdd'lstm_24/lstm_cell_24/MatMul_3:product:0%lstm_24/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_4Mullstm_24/zeros:output:0)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_5Mullstm_24/zeros:output:0)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_6Mullstm_24/zeros:output:0)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_7Mullstm_24/zeros:output:0)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/ReadVariableOpReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0y
(lstm_24/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_24/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_24/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_24/lstm_cell_24/strided_sliceStridedSlice+lstm_24/lstm_cell_24/ReadVariableOp:value:01lstm_24/lstm_cell_24/strided_slice/stack:output:03lstm_24/lstm_cell_24/strided_slice/stack_1:output:03lstm_24/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¦
lstm_24/lstm_cell_24/MatMul_4MatMullstm_24/lstm_cell_24/mul_4:z:0+lstm_24/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_24/lstm_cell_24/addAddV2%lstm_24/lstm_cell_24/BiasAdd:output:0'lstm_24/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_24/lstm_cell_24/SigmoidSigmoidlstm_24/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_24/lstm_cell_24/ReadVariableOp_1ReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_24/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_24/lstm_cell_24/strided_slice_1StridedSlice-lstm_24/lstm_cell_24/ReadVariableOp_1:value:03lstm_24/lstm_cell_24/strided_slice_1/stack:output:05lstm_24/lstm_cell_24/strided_slice_1/stack_1:output:05lstm_24/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¨
lstm_24/lstm_cell_24/MatMul_5MatMullstm_24/lstm_cell_24/mul_5:z:0-lstm_24/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_24/lstm_cell_24/add_1AddV2'lstm_24/lstm_cell_24/BiasAdd_1:output:0'lstm_24/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_1Sigmoidlstm_24/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_8Mul"lstm_24/lstm_cell_24/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_24/lstm_cell_24/ReadVariableOp_2ReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_24/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_24/lstm_cell_24/strided_slice_2StridedSlice-lstm_24/lstm_cell_24/ReadVariableOp_2:value:03lstm_24/lstm_cell_24/strided_slice_2/stack:output:05lstm_24/lstm_cell_24/strided_slice_2/stack_1:output:05lstm_24/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¨
lstm_24/lstm_cell_24/MatMul_6MatMullstm_24/lstm_cell_24/mul_6:z:0-lstm_24/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_24/lstm_cell_24/add_2AddV2'lstm_24/lstm_cell_24/BiasAdd_2:output:0'lstm_24/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_2Sigmoidlstm_24/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_9Mul lstm_24/lstm_cell_24/Sigmoid:y:0"lstm_24/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/add_3AddV2lstm_24/lstm_cell_24/mul_8:z:0lstm_24/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_24/lstm_cell_24/ReadVariableOp_3ReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_24/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_24/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_24/lstm_cell_24/strided_slice_3StridedSlice-lstm_24/lstm_cell_24/ReadVariableOp_3:value:03lstm_24/lstm_cell_24/strided_slice_3/stack:output:05lstm_24/lstm_cell_24/strided_slice_3/stack_1:output:05lstm_24/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¨
lstm_24/lstm_cell_24/MatMul_7MatMullstm_24/lstm_cell_24/mul_7:z:0-lstm_24/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_24/lstm_cell_24/add_4AddV2'lstm_24/lstm_cell_24/BiasAdd_3:output:0'lstm_24/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_3Sigmoidlstm_24/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_4Sigmoidlstm_24/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_10Mul"lstm_24/lstm_cell_24/Sigmoid_3:y:0"lstm_24/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_24/TensorArrayV2_1TensorListReserve.lstm_24/TensorArrayV2_1/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : è
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_24_lstm_cell_24_split_readvariableop_resource4lstm_24_lstm_cell_24_split_1_readvariableop_resource,lstm_24_lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_24_while_body_482791*%
condR
lstm_24_while_cond_482790*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ú
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_73/MatMulMatMul lstm_24/strided_slice_3:output:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_74/MatMulMatMuldense_73/BiasAdd:output:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_72/BiasAdd/ReadVariableOp"^dense_72/Tensordot/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp$^lstm_24/lstm_cell_24/ReadVariableOp&^lstm_24/lstm_cell_24/ReadVariableOp_1&^lstm_24/lstm_cell_24/ReadVariableOp_2&^lstm_24/lstm_cell_24/ReadVariableOp_3*^lstm_24/lstm_cell_24/split/ReadVariableOp,^lstm_24/lstm_cell_24/split_1/ReadVariableOp^lstm_24/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2F
!dense_72/Tensordot/ReadVariableOp!dense_72/Tensordot/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2J
#lstm_24/lstm_cell_24/ReadVariableOp#lstm_24/lstm_cell_24/ReadVariableOp2N
%lstm_24/lstm_cell_24/ReadVariableOp_1%lstm_24/lstm_cell_24/ReadVariableOp_12N
%lstm_24/lstm_cell_24/ReadVariableOp_2%lstm_24/lstm_cell_24/ReadVariableOp_22N
%lstm_24/lstm_cell_24/ReadVariableOp_3%lstm_24/lstm_cell_24/ReadVariableOp_32V
)lstm_24/lstm_cell_24/split/ReadVariableOp)lstm_24/lstm_cell_24/split/ReadVariableOp2Z
+lstm_24/lstm_cell_24/split_1/ReadVariableOp+lstm_24/lstm_cell_24/split_1/ReadVariableOp2
lstm_24/whilelstm_24/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
â
â
I__inference_sequential_23_layer_call_and_return_conditional_losses_483346

inputs<
*dense_72_tensordot_readvariableop_resource:6
(dense_72_biasadd_readvariableop_resource:D
2lstm_24_lstm_cell_24_split_readvariableop_resource:B
4lstm_24_lstm_cell_24_split_1_readvariableop_resource:>
,lstm_24_lstm_cell_24_readvariableop_resource:9
'dense_73_matmul_readvariableop_resource:6
(dense_73_biasadd_readvariableop_resource:9
'dense_74_matmul_readvariableop_resource:6
(dense_74_biasadd_readvariableop_resource:
identity¢dense_72/BiasAdd/ReadVariableOp¢!dense_72/Tensordot/ReadVariableOp¢dense_73/BiasAdd/ReadVariableOp¢dense_73/MatMul/ReadVariableOp¢dense_74/BiasAdd/ReadVariableOp¢dense_74/MatMul/ReadVariableOp¢#lstm_24/lstm_cell_24/ReadVariableOp¢%lstm_24/lstm_cell_24/ReadVariableOp_1¢%lstm_24/lstm_cell_24/ReadVariableOp_2¢%lstm_24/lstm_cell_24/ReadVariableOp_3¢)lstm_24/lstm_cell_24/split/ReadVariableOp¢+lstm_24/lstm_cell_24/split_1/ReadVariableOp¢lstm_24/while
!dense_72/Tensordot/ReadVariableOpReadVariableOp*dense_72_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0a
dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:h
dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       N
dense_72/Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:b
 dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ß
dense_72/Tensordot/GatherV2GatherV2!dense_72/Tensordot/Shape:output:0 dense_72/Tensordot/free:output:0)dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:d
"dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ã
dense_72/Tensordot/GatherV2_1GatherV2!dense_72/Tensordot/Shape:output:0 dense_72/Tensordot/axes:output:0+dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:b
dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
dense_72/Tensordot/ProdProd$dense_72/Tensordot/GatherV2:output:0!dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: d
dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
dense_72/Tensordot/Prod_1Prod&dense_72/Tensordot/GatherV2_1:output:0#dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: `
dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : À
dense_72/Tensordot/concatConcatV2 dense_72/Tensordot/free:output:0 dense_72/Tensordot/axes:output:0'dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
dense_72/Tensordot/stackPack dense_72/Tensordot/Prod:output:0"dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
dense_72/Tensordot/transpose	Transposeinputs"dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
dense_72/Tensordot/ReshapeReshape dense_72/Tensordot/transpose:y:0!dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¥
dense_72/Tensordot/MatMulMatMul#dense_72/Tensordot/Reshape:output:0)dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:b
 dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ë
dense_72/Tensordot/concat_1ConcatV2$dense_72/Tensordot/GatherV2:output:0#dense_72/Tensordot/Const_2:output:0)dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
dense_72/TensordotReshape#dense_72/Tensordot/MatMul:product:0$dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_72/BiasAdd/ReadVariableOpReadVariableOp(dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_72/BiasAddBiasAdddense_72/Tensordot:output:0'dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
lstm_24/ShapeShapedense_72/BiasAdd:output:0*
T0*
_output_shapes
:e
lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: g
lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:g
lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ù
lstm_24/strided_sliceStridedSlicelstm_24/Shape:output:0$lstm_24/strided_slice/stack:output:0&lstm_24/strided_slice/stack_1:output:0&lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskX
lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_24/zeros/packedPacklstm_24/strided_slice:output:0lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:X
lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_24/zerosFilllstm_24/zeros/packed:output:0lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :
lstm_24/zeros_1/packedPacklstm_24/strided_slice:output:0!lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Z
lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_24/zeros_1Filllstm_24/zeros_1/packed:output:0lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
lstm_24/transpose	Transposedense_72/BiasAdd:output:0lstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
lstm_24/Shape_1Shapelstm_24/transpose:y:0*
T0*
_output_shapes
:g
lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_24/strided_slice_1StridedSlicelstm_24/Shape_1:output:0&lstm_24/strided_slice_1/stack:output:0(lstm_24/strided_slice_1/stack_1:output:0(lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskn
#lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÌ
lstm_24/TensorArrayV2TensorListReserve,lstm_24/TensorArrayV2/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
=lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ø
/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_24/transpose:y:0Flstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒg
lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: i
lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:i
lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_24/strided_slice_2StridedSlicelstm_24/transpose:y:0&lstm_24/strided_slice_2/stack:output:0(lstm_24/strided_slice_2/stack_1:output:0(lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskt
$lstm_24/lstm_cell_24/ones_like/ShapeShape lstm_24/strided_slice_2:output:0*
T0*
_output_shapes
:i
$lstm_24/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
lstm_24/lstm_cell_24/ones_likeFill-lstm_24/lstm_cell_24/ones_like/Shape:output:0-lstm_24/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"lstm_24/lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 lstm_24/lstm_cell_24/dropout/MulMul'lstm_24/lstm_cell_24/ones_like:output:0+lstm_24/lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"lstm_24/lstm_cell_24/dropout/ShapeShape'lstm_24/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9lstm_24/lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform+lstm_24/lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+lstm_24/lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)lstm_24/lstm_cell_24/dropout/GreaterEqualGreaterEqualBlstm_24/lstm_cell_24/dropout/random_uniform/RandomUniform:output:04lstm_24/lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!lstm_24/lstm_cell_24/dropout/CastCast-lstm_24/lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"lstm_24/lstm_cell_24/dropout/Mul_1Mul$lstm_24/lstm_cell_24/dropout/Mul:z:0%lstm_24/lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm_24/lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?³
"lstm_24/lstm_cell_24/dropout_1/MulMul'lstm_24/lstm_cell_24/ones_like:output:0-lstm_24/lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
$lstm_24/lstm_cell_24/dropout_1/ShapeShape'lstm_24/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:º
;lstm_24/lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform-lstm_24/lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0r
-lstm_24/lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ë
+lstm_24/lstm_cell_24/dropout_1/GreaterEqualGreaterEqualDlstm_24/lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:06lstm_24/lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/dropout_1/CastCast/lstm_24/lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
$lstm_24/lstm_cell_24/dropout_1/Mul_1Mul&lstm_24/lstm_cell_24/dropout_1/Mul:z:0'lstm_24/lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm_24/lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?³
"lstm_24/lstm_cell_24/dropout_2/MulMul'lstm_24/lstm_cell_24/ones_like:output:0-lstm_24/lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
$lstm_24/lstm_cell_24/dropout_2/ShapeShape'lstm_24/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:º
;lstm_24/lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform-lstm_24/lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0r
-lstm_24/lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ë
+lstm_24/lstm_cell_24/dropout_2/GreaterEqualGreaterEqualDlstm_24/lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:06lstm_24/lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/dropout_2/CastCast/lstm_24/lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
$lstm_24/lstm_cell_24/dropout_2/Mul_1Mul&lstm_24/lstm_cell_24/dropout_2/Mul:z:0'lstm_24/lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm_24/lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?³
"lstm_24/lstm_cell_24/dropout_3/MulMul'lstm_24/lstm_cell_24/ones_like:output:0-lstm_24/lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
$lstm_24/lstm_cell_24/dropout_3/ShapeShape'lstm_24/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:º
;lstm_24/lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform-lstm_24/lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0r
-lstm_24/lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ë
+lstm_24/lstm_cell_24/dropout_3/GreaterEqualGreaterEqualDlstm_24/lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:06lstm_24/lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/dropout_3/CastCast/lstm_24/lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
$lstm_24/lstm_cell_24/dropout_3/Mul_1Mul&lstm_24/lstm_cell_24/dropout_3/Mul:z:0'lstm_24/lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
&lstm_24/lstm_cell_24/ones_like_1/ShapeShapelstm_24/zeros:output:0*
T0*
_output_shapes
:k
&lstm_24/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¼
 lstm_24/lstm_cell_24/ones_like_1Fill/lstm_24/lstm_cell_24/ones_like_1/Shape:output:0/lstm_24/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm_24/lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?µ
"lstm_24/lstm_cell_24/dropout_4/MulMul)lstm_24/lstm_cell_24/ones_like_1:output:0-lstm_24/lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
$lstm_24/lstm_cell_24/dropout_4/ShapeShape)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:º
;lstm_24/lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform-lstm_24/lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0r
-lstm_24/lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ë
+lstm_24/lstm_cell_24/dropout_4/GreaterEqualGreaterEqualDlstm_24/lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:06lstm_24/lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/dropout_4/CastCast/lstm_24/lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
$lstm_24/lstm_cell_24/dropout_4/Mul_1Mul&lstm_24/lstm_cell_24/dropout_4/Mul:z:0'lstm_24/lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm_24/lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?µ
"lstm_24/lstm_cell_24/dropout_5/MulMul)lstm_24/lstm_cell_24/ones_like_1:output:0-lstm_24/lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
$lstm_24/lstm_cell_24/dropout_5/ShapeShape)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:º
;lstm_24/lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform-lstm_24/lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0r
-lstm_24/lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ë
+lstm_24/lstm_cell_24/dropout_5/GreaterEqualGreaterEqualDlstm_24/lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:06lstm_24/lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/dropout_5/CastCast/lstm_24/lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
$lstm_24/lstm_cell_24/dropout_5/Mul_1Mul&lstm_24/lstm_cell_24/dropout_5/Mul:z:0'lstm_24/lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm_24/lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?µ
"lstm_24/lstm_cell_24/dropout_6/MulMul)lstm_24/lstm_cell_24/ones_like_1:output:0-lstm_24/lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
$lstm_24/lstm_cell_24/dropout_6/ShapeShape)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:º
;lstm_24/lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform-lstm_24/lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0r
-lstm_24/lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ë
+lstm_24/lstm_cell_24/dropout_6/GreaterEqualGreaterEqualDlstm_24/lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:06lstm_24/lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/dropout_6/CastCast/lstm_24/lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
$lstm_24/lstm_cell_24/dropout_6/Mul_1Mul&lstm_24/lstm_cell_24/dropout_6/Mul:z:0'lstm_24/lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
$lstm_24/lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?µ
"lstm_24/lstm_cell_24/dropout_7/MulMul)lstm_24/lstm_cell_24/ones_like_1:output:0-lstm_24/lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
$lstm_24/lstm_cell_24/dropout_7/ShapeShape)lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:º
;lstm_24/lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform-lstm_24/lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0r
-lstm_24/lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ë
+lstm_24/lstm_cell_24/dropout_7/GreaterEqualGreaterEqualDlstm_24/lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:06lstm_24/lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/dropout_7/CastCast/lstm_24/lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
$lstm_24/lstm_cell_24/dropout_7/Mul_1Mul&lstm_24/lstm_cell_24/dropout_7/Mul:z:0'lstm_24/lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mulMul lstm_24/strided_slice_2:output:0&lstm_24/lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_1Mul lstm_24/strided_slice_2:output:0(lstm_24/lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_2Mul lstm_24/strided_slice_2:output:0(lstm_24/lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_3Mul lstm_24/strided_slice_2:output:0(lstm_24/lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
)lstm_24/lstm_cell_24/split/ReadVariableOpReadVariableOp2lstm_24_lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Ý
lstm_24/lstm_cell_24/splitSplit-lstm_24/lstm_cell_24/split/split_dim:output:01lstm_24/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_24/lstm_cell_24/MatMulMatMullstm_24/lstm_cell_24/mul:z:0#lstm_24/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/MatMul_1MatMullstm_24/lstm_cell_24/mul_1:z:0#lstm_24/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/MatMul_2MatMullstm_24/lstm_cell_24/mul_2:z:0#lstm_24/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/MatMul_3MatMullstm_24/lstm_cell_24/mul_3:z:0#lstm_24/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&lstm_24/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
+lstm_24/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4lstm_24_lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0Ó
lstm_24/lstm_cell_24/split_1Split/lstm_24/lstm_cell_24/split_1/split_dim:output:03lstm_24/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split§
lstm_24/lstm_cell_24/BiasAddBiasAdd%lstm_24/lstm_cell_24/MatMul:product:0%lstm_24/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_24/lstm_cell_24/BiasAdd_1BiasAdd'lstm_24/lstm_cell_24/MatMul_1:product:0%lstm_24/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_24/lstm_cell_24/BiasAdd_2BiasAdd'lstm_24/lstm_cell_24/MatMul_2:product:0%lstm_24/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
lstm_24/lstm_cell_24/BiasAdd_3BiasAdd'lstm_24/lstm_cell_24/MatMul_3:product:0%lstm_24/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_4Mullstm_24/zeros:output:0(lstm_24/lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_5Mullstm_24/zeros:output:0(lstm_24/lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_6Mullstm_24/zeros:output:0(lstm_24/lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_7Mullstm_24/zeros:output:0(lstm_24/lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_24/lstm_cell_24/ReadVariableOpReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0y
(lstm_24/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        {
*lstm_24/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*lstm_24/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"lstm_24/lstm_cell_24/strided_sliceStridedSlice+lstm_24/lstm_cell_24/ReadVariableOp:value:01lstm_24/lstm_cell_24/strided_slice/stack:output:03lstm_24/lstm_cell_24/strided_slice/stack_1:output:03lstm_24/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¦
lstm_24/lstm_cell_24/MatMul_4MatMullstm_24/lstm_cell_24/mul_4:z:0+lstm_24/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
lstm_24/lstm_cell_24/addAddV2%lstm_24/lstm_cell_24/BiasAdd:output:0'lstm_24/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
lstm_24/lstm_cell_24/SigmoidSigmoidlstm_24/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_24/lstm_cell_24/ReadVariableOp_1ReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_24/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_24/lstm_cell_24/strided_slice_1StridedSlice-lstm_24/lstm_cell_24/ReadVariableOp_1:value:03lstm_24/lstm_cell_24/strided_slice_1/stack:output:05lstm_24/lstm_cell_24/strided_slice_1/stack_1:output:05lstm_24/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¨
lstm_24/lstm_cell_24/MatMul_5MatMullstm_24/lstm_cell_24/mul_5:z:0-lstm_24/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_24/lstm_cell_24/add_1AddV2'lstm_24/lstm_cell_24/BiasAdd_1:output:0'lstm_24/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_1Sigmoidlstm_24/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_8Mul"lstm_24/lstm_cell_24/Sigmoid_1:y:0lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_24/lstm_cell_24/ReadVariableOp_2ReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_24/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_24/lstm_cell_24/strided_slice_2StridedSlice-lstm_24/lstm_cell_24/ReadVariableOp_2:value:03lstm_24/lstm_cell_24/strided_slice_2/stack:output:05lstm_24/lstm_cell_24/strided_slice_2/stack_1:output:05lstm_24/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¨
lstm_24/lstm_cell_24/MatMul_6MatMullstm_24/lstm_cell_24/mul_6:z:0-lstm_24/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_24/lstm_cell_24/add_2AddV2'lstm_24/lstm_cell_24/BiasAdd_2:output:0'lstm_24/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_2Sigmoidlstm_24/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_9Mul lstm_24/lstm_cell_24/Sigmoid:y:0"lstm_24/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/add_3AddV2lstm_24/lstm_cell_24/mul_8:z:0lstm_24/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%lstm_24/lstm_cell_24/ReadVariableOp_3ReadVariableOp,lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0{
*lstm_24/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       }
,lstm_24/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        }
,lstm_24/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Þ
$lstm_24/lstm_cell_24/strided_slice_3StridedSlice-lstm_24/lstm_cell_24/ReadVariableOp_3:value:03lstm_24/lstm_cell_24/strided_slice_3/stack:output:05lstm_24/lstm_cell_24/strided_slice_3/stack_1:output:05lstm_24/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¨
lstm_24/lstm_cell_24/MatMul_7MatMullstm_24/lstm_cell_24/mul_7:z:0-lstm_24/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
lstm_24/lstm_cell_24/add_4AddV2'lstm_24/lstm_cell_24/BiasAdd_3:output:0'lstm_24/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_3Sigmoidlstm_24/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ{
lstm_24/lstm_cell_24/Sigmoid_4Sigmoidlstm_24/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/lstm_cell_24/mul_10Mul"lstm_24/lstm_cell_24/Sigmoid_3:y:0"lstm_24/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
%lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ð
lstm_24/TensorArrayV2_1TensorListReserve.lstm_24/TensorArrayV2_1/element_shape:output:0 lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒN
lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : k
 lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ\
lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : è
lstm_24/whileWhile#lstm_24/while/loop_counter:output:0)lstm_24/while/maximum_iterations:output:0lstm_24/time:output:0 lstm_24/TensorArrayV2_1:handle:0lstm_24/zeros:output:0lstm_24/zeros_1:output:0 lstm_24/strided_slice_1:output:0?lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:02lstm_24_lstm_cell_24_split_readvariableop_resource4lstm_24_lstm_cell_24_split_1_readvariableop_resource,lstm_24_lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *%
bodyR
lstm_24_while_body_483136*%
condR
lstm_24_while_cond_483135*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
8lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ú
*lstm_24/TensorArrayV2Stack/TensorListStackTensorListStacklstm_24/while:output:3Alstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0p
lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿi
lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: i
lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¯
lstm_24/strided_slice_3StridedSlice3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0&lstm_24/strided_slice_3/stack:output:0(lstm_24/strided_slice_3/stack_1:output:0(lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskm
lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          ®
lstm_24/transpose_1	Transpose3lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0!lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_73/MatMul/ReadVariableOpReadVariableOp'dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_73/MatMulMatMul lstm_24/strided_slice_3:output:0&dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_73/BiasAdd/ReadVariableOpReadVariableOp(dense_73_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_73/BiasAddBiasAdddense_73/MatMul:product:0'dense_73/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_74/MatMul/ReadVariableOpReadVariableOp'dense_74_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
dense_74/MatMulMatMuldense_73/BiasAdd:output:0&dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_74/BiasAdd/ReadVariableOpReadVariableOp(dense_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_74/BiasAddBiasAdddense_74/MatMul:product:0'dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
IdentityIdentitydense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_72/BiasAdd/ReadVariableOp"^dense_72/Tensordot/ReadVariableOp ^dense_73/BiasAdd/ReadVariableOp^dense_73/MatMul/ReadVariableOp ^dense_74/BiasAdd/ReadVariableOp^dense_74/MatMul/ReadVariableOp$^lstm_24/lstm_cell_24/ReadVariableOp&^lstm_24/lstm_cell_24/ReadVariableOp_1&^lstm_24/lstm_cell_24/ReadVariableOp_2&^lstm_24/lstm_cell_24/ReadVariableOp_3*^lstm_24/lstm_cell_24/split/ReadVariableOp,^lstm_24/lstm_cell_24/split_1/ReadVariableOp^lstm_24/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2B
dense_72/BiasAdd/ReadVariableOpdense_72/BiasAdd/ReadVariableOp2F
!dense_72/Tensordot/ReadVariableOp!dense_72/Tensordot/ReadVariableOp2B
dense_73/BiasAdd/ReadVariableOpdense_73/BiasAdd/ReadVariableOp2@
dense_73/MatMul/ReadVariableOpdense_73/MatMul/ReadVariableOp2B
dense_74/BiasAdd/ReadVariableOpdense_74/BiasAdd/ReadVariableOp2@
dense_74/MatMul/ReadVariableOpdense_74/MatMul/ReadVariableOp2J
#lstm_24/lstm_cell_24/ReadVariableOp#lstm_24/lstm_cell_24/ReadVariableOp2N
%lstm_24/lstm_cell_24/ReadVariableOp_1%lstm_24/lstm_cell_24/ReadVariableOp_12N
%lstm_24/lstm_cell_24/ReadVariableOp_2%lstm_24/lstm_cell_24/ReadVariableOp_22N
%lstm_24/lstm_cell_24/ReadVariableOp_3%lstm_24/lstm_cell_24/ReadVariableOp_32V
)lstm_24/lstm_cell_24/split/ReadVariableOp)lstm_24/lstm_cell_24/split/ReadVariableOp2Z
+lstm_24/lstm_cell_24/split_1/ReadVariableOp+lstm_24/lstm_cell_24/split_1/ReadVariableOp2
lstm_24/whilelstm_24/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ô
²
(__inference_lstm_24_layer_call_fn_483443

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_481972o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_484176
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_484176___redundant_placeholder04
0while_while_cond_484176___redundant_placeholder14
0while_while_cond_484176___redundant_placeholder24
0while_while_cond_484176___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
â	
Ø
$__inference_signature_wrapper_483371
dense_72_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalldense_72_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_481173o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_72_input
ÔÄ
	
while_body_482240
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_24_split_readvariableop_resource_0:B
4while_lstm_cell_24_split_1_readvariableop_resource_0:>
,while_lstm_cell_24_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_24_split_readvariableop_resource:@
2while_lstm_cell_24_split_1_readvariableop_resource:<
*while_lstm_cell_24_readvariableop_resource:¢!while/lstm_cell_24/ReadVariableOp¢#while/lstm_cell_24/ReadVariableOp_1¢#while/lstm_cell_24/ReadVariableOp_2¢#while/lstm_cell_24/ReadVariableOp_3¢'while/lstm_cell_24/split/ReadVariableOp¢)while/lstm_cell_24/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
"while/lstm_cell_24/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
while/lstm_cell_24/ones_likeFill+while/lstm_cell_24/ones_like/Shape:output:0+while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 while/lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?©
while/lstm_cell_24/dropout/MulMul%while/lstm_cell_24/ones_like:output:0)while/lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 while/lstm_cell_24/dropout/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:²
7while/lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0n
)while/lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ß
'while/lstm_cell_24/dropout/GreaterEqualGreaterEqual@while/lstm_cell_24/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/dropout/CastCast+while/lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 while/lstm_cell_24/dropout/Mul_1Mul"while/lstm_cell_24/dropout/Mul:z:0#while/lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_1/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_1/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_1/CastCast-while/lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_1/Mul_1Mul$while/lstm_cell_24/dropout_1/Mul:z:0%while/lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_2/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_2/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_2/CastCast-while/lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_2/Mul_1Mul$while/lstm_cell_24/dropout_2/Mul:z:0%while/lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_3/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_3/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_3/CastCast-while/lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_3/Mul_1Mul$while/lstm_cell_24/dropout_3/Mul:z:0%while/lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$while/lstm_cell_24/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
while/lstm_cell_24/ones_like_1Fill-while/lstm_cell_24/ones_like_1/Shape:output:0-while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_4/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_4/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_4/CastCast-while/lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_4/Mul_1Mul$while/lstm_cell_24/dropout_4/Mul:z:0%while/lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_5/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_5/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_5/CastCast-while/lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_5/Mul_1Mul$while/lstm_cell_24/dropout_5/Mul:z:0%while/lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_6/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_6/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_6/CastCast-while/lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_6/Mul_1Mul$while/lstm_cell_24/dropout_6/Mul:z:0%while/lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_7/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_7/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_7/CastCast-while/lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_7/Mul_1Mul$while/lstm_cell_24/dropout_7/Mul:z:0%while/lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell_24/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_24/split/ReadVariableOpReadVariableOp2while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0×
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
while/lstm_cell_24/MatMulMatMulwhile/lstm_cell_24/mul:z:0!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_1MatMulwhile/lstm_cell_24/mul_1:z:0!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_2MatMulwhile/lstm_cell_24/mul_2:z:0!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_3MatMulwhile/lstm_cell_24/mul_3:z:0!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0Í
while/lstm_cell_24/split_1Split-while/lstm_cell_24/split_1/split_dim:output:01while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¡
while/lstm_cell_24/BiasAddBiasAdd#while/lstm_cell_24/MatMul:product:0#while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_1BiasAdd%while/lstm_cell_24/MatMul_1:product:0#while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_2BiasAdd%while/lstm_cell_24/MatMul_2:product:0#while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_3BiasAdd%while/lstm_cell_24/MatMul_3:product:0#while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_4Mulwhile_placeholder_2&while/lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_5Mulwhile_placeholder_2&while/lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_6Mulwhile_placeholder_2&while/lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_7Mulwhile_placeholder_2&while/lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/ReadVariableOpReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 while/lstm_cell_24/strided_sliceStridedSlice)while/lstm_cell_24/ReadVariableOp:value:0/while/lstm_cell_24/strided_slice/stack:output:01while/lstm_cell_24/strided_slice/stack_1:output:01while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask 
while/lstm_cell_24/MatMul_4MatMulwhile/lstm_cell_24/mul_4:z:0)while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/addAddV2#while/lstm_cell_24/BiasAdd:output:0%while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_24/SigmoidSigmoidwhile/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_1ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_1StridedSlice+while/lstm_cell_24/ReadVariableOp_1:value:01while/lstm_cell_24/strided_slice_1/stack:output:03while/lstm_cell_24/strided_slice_1/stack_1:output:03while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_5MatMulwhile/lstm_cell_24/mul_5:z:0+while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_1AddV2%while/lstm_cell_24/BiasAdd_1:output:0%while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_1Sigmoidwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_8Mul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_2ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_2StridedSlice+while/lstm_cell_24/ReadVariableOp_2:value:01while/lstm_cell_24/strided_slice_2/stack:output:03while/lstm_cell_24/strided_slice_2/stack_1:output:03while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_6MatMulwhile/lstm_cell_24/mul_6:z:0+while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_2AddV2%while/lstm_cell_24/BiasAdd_2:output:0%while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_2Sigmoidwhile/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_9Mulwhile/lstm_cell_24/Sigmoid:y:0 while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/add_3AddV2while/lstm_cell_24/mul_8:z:0while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_3ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_3StridedSlice+while/lstm_cell_24/ReadVariableOp_3:value:01while/lstm_cell_24/strided_slice_3/stack:output:03while/lstm_cell_24/strided_slice_3/stack_1:output:03while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_7MatMulwhile/lstm_cell_24/mul_7:z:0+while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_4AddV2%while/lstm_cell_24/BiasAdd_3:output:0%while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_3Sigmoidwhile/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_4Sigmoidwhile/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_10Mul while/lstm_cell_24/Sigmoid_3:y:0 while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_24/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_24/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸

while/NoOpNoOp"^while/lstm_cell_24/ReadVariableOp$^while/lstm_cell_24/ReadVariableOp_1$^while/lstm_cell_24/ReadVariableOp_2$^while/lstm_cell_24/ReadVariableOp_3(^while/lstm_cell_24/split/ReadVariableOp*^while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_24_readvariableop_resource,while_lstm_cell_24_readvariableop_resource_0"j
2while_lstm_cell_24_split_1_readvariableop_resource4while_lstm_cell_24_split_1_readvariableop_resource_0"f
0while_lstm_cell_24_split_readvariableop_resource2while_lstm_cell_24_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2F
!while/lstm_cell_24/ReadVariableOp!while/lstm_cell_24/ReadVariableOp2J
#while/lstm_cell_24/ReadVariableOp_1#while/lstm_cell_24/ReadVariableOp_12J
#while/lstm_cell_24/ReadVariableOp_2#while/lstm_cell_24/ReadVariableOp_22J
#while/lstm_cell_24/ReadVariableOp_3#while/lstm_cell_24/ReadVariableOp_32R
'while/lstm_cell_24/split/ReadVariableOp'while/lstm_cell_24/split/ReadVariableOp2V
)while/lstm_cell_24/split_1/ReadVariableOp)while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 

´
(__inference_lstm_24_layer_call_fn_483432
inputs_0
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_481678o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ÔÄ
	
while_body_483870
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_24_split_readvariableop_resource_0:B
4while_lstm_cell_24_split_1_readvariableop_resource_0:>
,while_lstm_cell_24_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_24_split_readvariableop_resource:@
2while_lstm_cell_24_split_1_readvariableop_resource:<
*while_lstm_cell_24_readvariableop_resource:¢!while/lstm_cell_24/ReadVariableOp¢#while/lstm_cell_24/ReadVariableOp_1¢#while/lstm_cell_24/ReadVariableOp_2¢#while/lstm_cell_24/ReadVariableOp_3¢'while/lstm_cell_24/split/ReadVariableOp¢)while/lstm_cell_24/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
"while/lstm_cell_24/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
while/lstm_cell_24/ones_likeFill+while/lstm_cell_24/ones_like/Shape:output:0+while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
 while/lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?©
while/lstm_cell_24/dropout/MulMul%while/lstm_cell_24/ones_like:output:0)while/lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
 while/lstm_cell_24/dropout/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:²
7while/lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform)while/lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0n
)while/lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ß
'while/lstm_cell_24/dropout/GreaterEqualGreaterEqual@while/lstm_cell_24/dropout/random_uniform/RandomUniform:output:02while/lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/dropout/CastCast+while/lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
 while/lstm_cell_24/dropout/Mul_1Mul"while/lstm_cell_24/dropout/Mul:z:0#while/lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_1/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_1/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_1/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_1/CastCast-while/lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_1/Mul_1Mul$while/lstm_cell_24/dropout_1/Mul:z:0%while/lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_2/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_2/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_2/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_2/CastCast-while/lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_2/Mul_1Mul$while/lstm_cell_24/dropout_2/Mul:z:0%while/lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?­
 while/lstm_cell_24/dropout_3/MulMul%while/lstm_cell_24/ones_like:output:0+while/lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
"while/lstm_cell_24/dropout_3/ShapeShape%while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_3/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_3/CastCast-while/lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_3/Mul_1Mul$while/lstm_cell_24/dropout_3/Mul:z:0%while/lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$while/lstm_cell_24/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
while/lstm_cell_24/ones_like_1Fill-while/lstm_cell_24/ones_like_1/Shape:output:0-while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_4/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_4/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_4/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_4/CastCast-while/lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_4/Mul_1Mul$while/lstm_cell_24/dropout_4/Mul:z:0%while/lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_5/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_5/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_5/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_5/CastCast-while/lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_5/Mul_1Mul$while/lstm_cell_24/dropout_5/Mul:z:0%while/lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_6/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_6/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_6/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_6/CastCast-while/lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_6/Mul_1Mul$while/lstm_cell_24/dropout_6/Mul:z:0%while/lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
"while/lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?¯
 while/lstm_cell_24/dropout_7/MulMul'while/lstm_cell_24/ones_like_1:output:0+while/lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
"while/lstm_cell_24/dropout_7/ShapeShape'while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:¶
9while/lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform+while/lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0p
+while/lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=å
)while/lstm_cell_24/dropout_7/GreaterEqualGreaterEqualBwhile/lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:04while/lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/dropout_7/CastCast-while/lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
"while/lstm_cell_24/dropout_7/Mul_1Mul$while/lstm_cell_24/dropout_7/Mul:z:0%while/lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
while/lstm_cell_24/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0$while/lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
while/lstm_cell_24/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0&while/lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_24/split/ReadVariableOpReadVariableOp2while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0×
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
while/lstm_cell_24/MatMulMatMulwhile/lstm_cell_24/mul:z:0!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_1MatMulwhile/lstm_cell_24/mul_1:z:0!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_2MatMulwhile/lstm_cell_24/mul_2:z:0!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_3MatMulwhile/lstm_cell_24/mul_3:z:0!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0Í
while/lstm_cell_24/split_1Split-while/lstm_cell_24/split_1/split_dim:output:01while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¡
while/lstm_cell_24/BiasAddBiasAdd#while/lstm_cell_24/MatMul:product:0#while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_1BiasAdd%while/lstm_cell_24/MatMul_1:product:0#while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_2BiasAdd%while/lstm_cell_24/MatMul_2:product:0#while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_3BiasAdd%while/lstm_cell_24/MatMul_3:product:0#while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_4Mulwhile_placeholder_2&while/lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_5Mulwhile_placeholder_2&while/lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_6Mulwhile_placeholder_2&while/lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_7Mulwhile_placeholder_2&while/lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/ReadVariableOpReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 while/lstm_cell_24/strided_sliceStridedSlice)while/lstm_cell_24/ReadVariableOp:value:0/while/lstm_cell_24/strided_slice/stack:output:01while/lstm_cell_24/strided_slice/stack_1:output:01while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask 
while/lstm_cell_24/MatMul_4MatMulwhile/lstm_cell_24/mul_4:z:0)while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/addAddV2#while/lstm_cell_24/BiasAdd:output:0%while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_24/SigmoidSigmoidwhile/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_1ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_1StridedSlice+while/lstm_cell_24/ReadVariableOp_1:value:01while/lstm_cell_24/strided_slice_1/stack:output:03while/lstm_cell_24/strided_slice_1/stack_1:output:03while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_5MatMulwhile/lstm_cell_24/mul_5:z:0+while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_1AddV2%while/lstm_cell_24/BiasAdd_1:output:0%while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_1Sigmoidwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_8Mul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_2ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_2StridedSlice+while/lstm_cell_24/ReadVariableOp_2:value:01while/lstm_cell_24/strided_slice_2/stack:output:03while/lstm_cell_24/strided_slice_2/stack_1:output:03while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_6MatMulwhile/lstm_cell_24/mul_6:z:0+while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_2AddV2%while/lstm_cell_24/BiasAdd_2:output:0%while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_2Sigmoidwhile/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_9Mulwhile/lstm_cell_24/Sigmoid:y:0 while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/add_3AddV2while/lstm_cell_24/mul_8:z:0while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_3ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_3StridedSlice+while/lstm_cell_24/ReadVariableOp_3:value:01while/lstm_cell_24/strided_slice_3/stack:output:03while/lstm_cell_24/strided_slice_3/stack_1:output:03while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_7MatMulwhile/lstm_cell_24/mul_7:z:0+while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_4AddV2%while/lstm_cell_24/BiasAdd_3:output:0%while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_3Sigmoidwhile/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_4Sigmoidwhile/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_10Mul while/lstm_cell_24/Sigmoid_3:y:0 while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_24/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_24/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸

while/NoOpNoOp"^while/lstm_cell_24/ReadVariableOp$^while/lstm_cell_24/ReadVariableOp_1$^while/lstm_cell_24/ReadVariableOp_2$^while/lstm_cell_24/ReadVariableOp_3(^while/lstm_cell_24/split/ReadVariableOp*^while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_24_readvariableop_resource,while_lstm_cell_24_readvariableop_resource_0"j
2while_lstm_cell_24_split_1_readvariableop_resource4while_lstm_cell_24_split_1_readvariableop_resource_0"f
0while_lstm_cell_24_split_readvariableop_resource2while_lstm_cell_24_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2F
!while/lstm_cell_24/ReadVariableOp!while/lstm_cell_24/ReadVariableOp2J
#while/lstm_cell_24/ReadVariableOp_1#while/lstm_cell_24/ReadVariableOp_12J
#while/lstm_cell_24/ReadVariableOp_2#while/lstm_cell_24/ReadVariableOp_22J
#while/lstm_cell_24/ReadVariableOp_3#while/lstm_cell_24/ReadVariableOp_32R
'while/lstm_cell_24/split/ReadVariableOp'while/lstm_cell_24/split/ReadVariableOp2V
)while/lstm_cell_24/split_1/ReadVariableOp)while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_483869
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_483869___redundant_placeholder04
0while_while_cond_483869___redundant_placeholder14
0while_while_cond_483869___redundant_placeholder24
0while_while_cond_483869___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
µL
è
__inference__traced_save_485113
file_prefix.
*savev2_dense_72_kernel_read_readvariableop,
(savev2_dense_72_bias_read_readvariableop.
*savev2_dense_73_kernel_read_readvariableop,
(savev2_dense_73_bias_read_readvariableop.
*savev2_dense_74_kernel_read_readvariableop,
(savev2_dense_74_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_lstm_24_lstm_cell_24_kernel_read_readvariableopD
@savev2_lstm_24_lstm_cell_24_recurrent_kernel_read_readvariableop8
4savev2_lstm_24_lstm_cell_24_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop5
1savev2_adam_dense_72_kernel_m_read_readvariableop3
/savev2_adam_dense_72_bias_m_read_readvariableop5
1savev2_adam_dense_73_kernel_m_read_readvariableop3
/savev2_adam_dense_73_bias_m_read_readvariableop5
1savev2_adam_dense_74_kernel_m_read_readvariableop3
/savev2_adam_dense_74_bias_m_read_readvariableopA
=savev2_adam_lstm_24_lstm_cell_24_kernel_m_read_readvariableopK
Gsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_m_read_readvariableop?
;savev2_adam_lstm_24_lstm_cell_24_bias_m_read_readvariableop5
1savev2_adam_dense_72_kernel_v_read_readvariableop3
/savev2_adam_dense_72_bias_v_read_readvariableop5
1savev2_adam_dense_73_kernel_v_read_readvariableop3
/savev2_adam_dense_73_bias_v_read_readvariableop5
1savev2_adam_dense_74_kernel_v_read_readvariableop3
/savev2_adam_dense_74_bias_v_read_readvariableopA
=savev2_adam_lstm_24_lstm_cell_24_kernel_v_read_readvariableopK
Gsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_v_read_readvariableop?
;savev2_adam_lstm_24_lstm_cell_24_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*®
value¤B¡%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH·
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B À
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_dense_72_kernel_read_readvariableop(savev2_dense_72_bias_read_readvariableop*savev2_dense_73_kernel_read_readvariableop(savev2_dense_73_bias_read_readvariableop*savev2_dense_74_kernel_read_readvariableop(savev2_dense_74_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_lstm_24_lstm_cell_24_kernel_read_readvariableop@savev2_lstm_24_lstm_cell_24_recurrent_kernel_read_readvariableop4savev2_lstm_24_lstm_cell_24_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop1savev2_adam_dense_72_kernel_m_read_readvariableop/savev2_adam_dense_72_bias_m_read_readvariableop1savev2_adam_dense_73_kernel_m_read_readvariableop/savev2_adam_dense_73_bias_m_read_readvariableop1savev2_adam_dense_74_kernel_m_read_readvariableop/savev2_adam_dense_74_bias_m_read_readvariableop=savev2_adam_lstm_24_lstm_cell_24_kernel_m_read_readvariableopGsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_m_read_readvariableop;savev2_adam_lstm_24_lstm_cell_24_bias_m_read_readvariableop1savev2_adam_dense_72_kernel_v_read_readvariableop/savev2_adam_dense_72_bias_v_read_readvariableop1savev2_adam_dense_73_kernel_v_read_readvariableop/savev2_adam_dense_73_bias_v_read_readvariableop1savev2_adam_dense_74_kernel_v_read_readvariableop/savev2_adam_dense_74_bias_v_read_readvariableop=savev2_adam_lstm_24_lstm_cell_24_kernel_v_read_readvariableopGsavev2_adam_lstm_24_lstm_cell_24_recurrent_kernel_v_read_readvariableop;savev2_adam_lstm_24_lstm_cell_24_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *3
dtypes)
'2%	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*
_input_shapes÷
ô: ::::::: : : : : :::: : : : ::::::::::::::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 
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

:: 
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

:: 
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
µ
Ã
while_cond_481837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_481837___redundant_placeholder04
0while_while_cond_481837___redundant_placeholder14
0while_while_cond_481837___redundant_placeholder24
0while_while_cond_481837___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ç	
õ
D__inference_dense_73_layer_call_and_return_conditional_losses_481990

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ç	
õ
D__inference_dense_73_layer_call_and_return_conditional_losses_484701

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸ê
ö

!__inference__wrapped_model_481173
dense_72_inputJ
8sequential_23_dense_72_tensordot_readvariableop_resource:D
6sequential_23_dense_72_biasadd_readvariableop_resource:R
@sequential_23_lstm_24_lstm_cell_24_split_readvariableop_resource:P
Bsequential_23_lstm_24_lstm_cell_24_split_1_readvariableop_resource:L
:sequential_23_lstm_24_lstm_cell_24_readvariableop_resource:G
5sequential_23_dense_73_matmul_readvariableop_resource:D
6sequential_23_dense_73_biasadd_readvariableop_resource:G
5sequential_23_dense_74_matmul_readvariableop_resource:D
6sequential_23_dense_74_biasadd_readvariableop_resource:
identity¢-sequential_23/dense_72/BiasAdd/ReadVariableOp¢/sequential_23/dense_72/Tensordot/ReadVariableOp¢-sequential_23/dense_73/BiasAdd/ReadVariableOp¢,sequential_23/dense_73/MatMul/ReadVariableOp¢-sequential_23/dense_74/BiasAdd/ReadVariableOp¢,sequential_23/dense_74/MatMul/ReadVariableOp¢1sequential_23/lstm_24/lstm_cell_24/ReadVariableOp¢3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_1¢3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_2¢3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_3¢7sequential_23/lstm_24/lstm_cell_24/split/ReadVariableOp¢9sequential_23/lstm_24/lstm_cell_24/split_1/ReadVariableOp¢sequential_23/lstm_24/while¨
/sequential_23/dense_72/Tensordot/ReadVariableOpReadVariableOp8sequential_23_dense_72_tensordot_readvariableop_resource*
_output_shapes

:*
dtype0o
%sequential_23/dense_72/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:v
%sequential_23/dense_72/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       d
&sequential_23/dense_72/Tensordot/ShapeShapedense_72_input*
T0*
_output_shapes
:p
.sequential_23/dense_72/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)sequential_23/dense_72/Tensordot/GatherV2GatherV2/sequential_23/dense_72/Tensordot/Shape:output:0.sequential_23/dense_72/Tensordot/free:output:07sequential_23/dense_72/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:r
0sequential_23/dense_72/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
+sequential_23/dense_72/Tensordot/GatherV2_1GatherV2/sequential_23/dense_72/Tensordot/Shape:output:0.sequential_23/dense_72/Tensordot/axes:output:09sequential_23/dense_72/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
&sequential_23/dense_72/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ³
%sequential_23/dense_72/Tensordot/ProdProd2sequential_23/dense_72/Tensordot/GatherV2:output:0/sequential_23/dense_72/Tensordot/Const:output:0*
T0*
_output_shapes
: r
(sequential_23/dense_72/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ¹
'sequential_23/dense_72/Tensordot/Prod_1Prod4sequential_23/dense_72/Tensordot/GatherV2_1:output:01sequential_23/dense_72/Tensordot/Const_1:output:0*
T0*
_output_shapes
: n
,sequential_23/dense_72/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ø
'sequential_23/dense_72/Tensordot/concatConcatV2.sequential_23/dense_72/Tensordot/free:output:0.sequential_23/dense_72/Tensordot/axes:output:05sequential_23/dense_72/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¾
&sequential_23/dense_72/Tensordot/stackPack.sequential_23/dense_72/Tensordot/Prod:output:00sequential_23/dense_72/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:¯
*sequential_23/dense_72/Tensordot/transpose	Transposedense_72_input0sequential_23/dense_72/Tensordot/concat:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿÏ
(sequential_23/dense_72/Tensordot/ReshapeReshape.sequential_23/dense_72/Tensordot/transpose:y:0/sequential_23/dense_72/Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÏ
'sequential_23/dense_72/Tensordot/MatMulMatMul1sequential_23/dense_72/Tensordot/Reshape:output:07sequential_23/dense_72/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
(sequential_23/dense_72/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:p
.sequential_23/dense_72/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
)sequential_23/dense_72/Tensordot/concat_1ConcatV22sequential_23/dense_72/Tensordot/GatherV2:output:01sequential_23/dense_72/Tensordot/Const_2:output:07sequential_23/dense_72/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:È
 sequential_23/dense_72/TensordotReshape1sequential_23/dense_72/Tensordot/MatMul:product:02sequential_23/dense_72/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_23/dense_72/BiasAdd/ReadVariableOpReadVariableOp6sequential_23_dense_72_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Á
sequential_23/dense_72/BiasAddBiasAdd)sequential_23/dense_72/Tensordot:output:05sequential_23/dense_72/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
sequential_23/lstm_24/ShapeShape'sequential_23/dense_72/BiasAdd:output:0*
T0*
_output_shapes
:s
)sequential_23/lstm_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_23/lstm_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_23/lstm_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_23/lstm_24/strided_sliceStridedSlice$sequential_23/lstm_24/Shape:output:02sequential_23/lstm_24/strided_slice/stack:output:04sequential_23/lstm_24/strided_slice/stack_1:output:04sequential_23/lstm_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
$sequential_23/lstm_24/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :µ
"sequential_23/lstm_24/zeros/packedPack,sequential_23/lstm_24/strided_slice:output:0-sequential_23/lstm_24/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_23/lstm_24/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_23/lstm_24/zerosFill+sequential_23/lstm_24/zeros/packed:output:0*sequential_23/lstm_24/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
&sequential_23/lstm_24/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :¹
$sequential_23/lstm_24/zeros_1/packedPack,sequential_23/lstm_24/strided_slice:output:0/sequential_23/lstm_24/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:h
#sequential_23/lstm_24/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ´
sequential_23/lstm_24/zeros_1Fill-sequential_23/lstm_24/zeros_1/packed:output:0,sequential_23/lstm_24/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
$sequential_23/lstm_24/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          º
sequential_23/lstm_24/transpose	Transpose'sequential_23/dense_72/BiasAdd:output:0-sequential_23/lstm_24/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
sequential_23/lstm_24/Shape_1Shape#sequential_23/lstm_24/transpose:y:0*
T0*
_output_shapes
:u
+sequential_23/lstm_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_23/lstm_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:É
%sequential_23/lstm_24/strided_slice_1StridedSlice&sequential_23/lstm_24/Shape_1:output:04sequential_23/lstm_24/strided_slice_1/stack:output:06sequential_23/lstm_24/strided_slice_1/stack_1:output:06sequential_23/lstm_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask|
1sequential_23/lstm_24/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿö
#sequential_23/lstm_24/TensorArrayV2TensorListReserve:sequential_23/lstm_24/TensorArrayV2/element_shape:output:0.sequential_23/lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Ksequential_23/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¢
=sequential_23/lstm_24/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor#sequential_23/lstm_24/transpose:y:0Tsequential_23/lstm_24/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒu
+sequential_23/lstm_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:w
-sequential_23/lstm_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:×
%sequential_23/lstm_24/strided_slice_2StridedSlice#sequential_23/lstm_24/transpose:y:04sequential_23/lstm_24/strided_slice_2/stack:output:06sequential_23/lstm_24/strided_slice_2/stack_1:output:06sequential_23/lstm_24/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
2sequential_23/lstm_24/lstm_cell_24/ones_like/ShapeShape.sequential_23/lstm_24/strided_slice_2:output:0*
T0*
_output_shapes
:w
2sequential_23/lstm_24/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?à
,sequential_23/lstm_24/lstm_cell_24/ones_likeFill;sequential_23/lstm_24/lstm_cell_24/ones_like/Shape:output:0;sequential_23/lstm_24/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
4sequential_23/lstm_24/lstm_cell_24/ones_like_1/ShapeShape$sequential_23/lstm_24/zeros:output:0*
T0*
_output_shapes
:y
4sequential_23/lstm_24/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?æ
.sequential_23/lstm_24/lstm_cell_24/ones_like_1Fill=sequential_23/lstm_24/lstm_cell_24/ones_like_1/Shape:output:0=sequential_23/lstm_24/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
&sequential_23/lstm_24/lstm_cell_24/mulMul.sequential_23/lstm_24/strided_slice_2:output:05sequential_23/lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(sequential_23/lstm_24/lstm_cell_24/mul_1Mul.sequential_23/lstm_24/strided_slice_2:output:05sequential_23/lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(sequential_23/lstm_24/lstm_cell_24/mul_2Mul.sequential_23/lstm_24/strided_slice_2:output:05sequential_23/lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
(sequential_23/lstm_24/lstm_cell_24/mul_3Mul.sequential_23/lstm_24/strided_slice_2:output:05sequential_23/lstm_24/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
2sequential_23/lstm_24/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¸
7sequential_23/lstm_24/lstm_cell_24/split/ReadVariableOpReadVariableOp@sequential_23_lstm_24_lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0
(sequential_23/lstm_24/lstm_cell_24/splitSplit;sequential_23/lstm_24/lstm_cell_24/split/split_dim:output:0?sequential_23/lstm_24/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitÄ
)sequential_23/lstm_24/lstm_cell_24/MatMulMatMul*sequential_23/lstm_24/lstm_cell_24/mul:z:01sequential_23/lstm_24/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
+sequential_23/lstm_24/lstm_cell_24/MatMul_1MatMul,sequential_23/lstm_24/lstm_cell_24/mul_1:z:01sequential_23/lstm_24/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
+sequential_23/lstm_24/lstm_cell_24/MatMul_2MatMul,sequential_23/lstm_24/lstm_cell_24/mul_2:z:01sequential_23/lstm_24/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÈ
+sequential_23/lstm_24/lstm_cell_24/MatMul_3MatMul,sequential_23/lstm_24/lstm_cell_24/mul_3:z:01sequential_23/lstm_24/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
4sequential_23/lstm_24/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ¸
9sequential_23/lstm_24/lstm_cell_24/split_1/ReadVariableOpReadVariableOpBsequential_23_lstm_24_lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0ý
*sequential_23/lstm_24/lstm_cell_24/split_1Split=sequential_23/lstm_24/lstm_cell_24/split_1/split_dim:output:0Asequential_23/lstm_24/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitÑ
*sequential_23/lstm_24/lstm_cell_24/BiasAddBiasAdd3sequential_23/lstm_24/lstm_cell_24/MatMul:product:03sequential_23/lstm_24/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
,sequential_23/lstm_24/lstm_cell_24/BiasAdd_1BiasAdd5sequential_23/lstm_24/lstm_cell_24/MatMul_1:product:03sequential_23/lstm_24/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
,sequential_23/lstm_24/lstm_cell_24/BiasAdd_2BiasAdd5sequential_23/lstm_24/lstm_cell_24/MatMul_2:product:03sequential_23/lstm_24/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
,sequential_23/lstm_24/lstm_cell_24/BiasAdd_3BiasAdd5sequential_23/lstm_24/lstm_cell_24/MatMul_3:product:03sequential_23/lstm_24/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
(sequential_23/lstm_24/lstm_cell_24/mul_4Mul$sequential_23/lstm_24/zeros:output:07sequential_23/lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
(sequential_23/lstm_24/lstm_cell_24/mul_5Mul$sequential_23/lstm_24/zeros:output:07sequential_23/lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
(sequential_23/lstm_24/lstm_cell_24/mul_6Mul$sequential_23/lstm_24/zeros:output:07sequential_23/lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
(sequential_23/lstm_24/lstm_cell_24/mul_7Mul$sequential_23/lstm_24/zeros:output:07sequential_23/lstm_24/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¬
1sequential_23/lstm_24/lstm_cell_24/ReadVariableOpReadVariableOp:sequential_23_lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0
6sequential_23/lstm_24/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
8sequential_23/lstm_24/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
8sequential_23/lstm_24/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      
0sequential_23/lstm_24/lstm_cell_24/strided_sliceStridedSlice9sequential_23/lstm_24/lstm_cell_24/ReadVariableOp:value:0?sequential_23/lstm_24/lstm_cell_24/strided_slice/stack:output:0Asequential_23/lstm_24/lstm_cell_24/strided_slice/stack_1:output:0Asequential_23/lstm_24/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskÐ
+sequential_23/lstm_24/lstm_cell_24/MatMul_4MatMul,sequential_23/lstm_24/lstm_cell_24/mul_4:z:09sequential_23/lstm_24/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÍ
&sequential_23/lstm_24/lstm_cell_24/addAddV23sequential_23/lstm_24/lstm_cell_24/BiasAdd:output:05sequential_23/lstm_24/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*sequential_23/lstm_24/lstm_cell_24/SigmoidSigmoid*sequential_23/lstm_24/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_1ReadVariableOp:sequential_23_lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0
8sequential_23/lstm_24/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
:sequential_23/lstm_24/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
:sequential_23/lstm_24/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
2sequential_23/lstm_24/lstm_cell_24/strided_slice_1StridedSlice;sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_1:value:0Asequential_23/lstm_24/lstm_cell_24/strided_slice_1/stack:output:0Csequential_23/lstm_24/lstm_cell_24/strided_slice_1/stack_1:output:0Csequential_23/lstm_24/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskÒ
+sequential_23/lstm_24/lstm_cell_24/MatMul_5MatMul,sequential_23/lstm_24/lstm_cell_24/mul_5:z:0;sequential_23/lstm_24/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
(sequential_23/lstm_24/lstm_cell_24/add_1AddV25sequential_23/lstm_24/lstm_cell_24/BiasAdd_1:output:05sequential_23/lstm_24/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_23/lstm_24/lstm_cell_24/Sigmoid_1Sigmoid,sequential_23/lstm_24/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
(sequential_23/lstm_24/lstm_cell_24/mul_8Mul0sequential_23/lstm_24/lstm_cell_24/Sigmoid_1:y:0&sequential_23/lstm_24/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_2ReadVariableOp:sequential_23_lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0
8sequential_23/lstm_24/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
:sequential_23/lstm_24/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
:sequential_23/lstm_24/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
2sequential_23/lstm_24/lstm_cell_24/strided_slice_2StridedSlice;sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_2:value:0Asequential_23/lstm_24/lstm_cell_24/strided_slice_2/stack:output:0Csequential_23/lstm_24/lstm_cell_24/strided_slice_2/stack_1:output:0Csequential_23/lstm_24/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskÒ
+sequential_23/lstm_24/lstm_cell_24/MatMul_6MatMul,sequential_23/lstm_24/lstm_cell_24/mul_6:z:0;sequential_23/lstm_24/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
(sequential_23/lstm_24/lstm_cell_24/add_2AddV25sequential_23/lstm_24/lstm_cell_24/BiasAdd_2:output:05sequential_23/lstm_24/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_23/lstm_24/lstm_cell_24/Sigmoid_2Sigmoid,sequential_23/lstm_24/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
(sequential_23/lstm_24/lstm_cell_24/mul_9Mul.sequential_23/lstm_24/lstm_cell_24/Sigmoid:y:00sequential_23/lstm_24/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
(sequential_23/lstm_24/lstm_cell_24/add_3AddV2,sequential_23/lstm_24/lstm_cell_24/mul_8:z:0,sequential_23/lstm_24/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_3ReadVariableOp:sequential_23_lstm_24_lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0
8sequential_23/lstm_24/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
:sequential_23/lstm_24/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
:sequential_23/lstm_24/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¤
2sequential_23/lstm_24/lstm_cell_24/strided_slice_3StridedSlice;sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_3:value:0Asequential_23/lstm_24/lstm_cell_24/strided_slice_3/stack:output:0Csequential_23/lstm_24/lstm_cell_24/strided_slice_3/stack_1:output:0Csequential_23/lstm_24/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskÒ
+sequential_23/lstm_24/lstm_cell_24/MatMul_7MatMul,sequential_23/lstm_24/lstm_cell_24/mul_7:z:0;sequential_23/lstm_24/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
(sequential_23/lstm_24/lstm_cell_24/add_4AddV25sequential_23/lstm_24/lstm_cell_24/BiasAdd_3:output:05sequential_23/lstm_24/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_23/lstm_24/lstm_cell_24/Sigmoid_3Sigmoid,sequential_23/lstm_24/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_23/lstm_24/lstm_cell_24/Sigmoid_4Sigmoid,sequential_23/lstm_24/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
)sequential_23/lstm_24/lstm_cell_24/mul_10Mul0sequential_23/lstm_24/lstm_cell_24/Sigmoid_3:y:00sequential_23/lstm_24/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
3sequential_23/lstm_24/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ú
%sequential_23/lstm_24/TensorArrayV2_1TensorListReserve<sequential_23/lstm_24/TensorArrayV2_1/element_shape:output:0.sequential_23/lstm_24/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ\
sequential_23/lstm_24/timeConst*
_output_shapes
: *
dtype0*
value	B : y
.sequential_23/lstm_24/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿj
(sequential_23/lstm_24/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ¬
sequential_23/lstm_24/whileWhile1sequential_23/lstm_24/while/loop_counter:output:07sequential_23/lstm_24/while/maximum_iterations:output:0#sequential_23/lstm_24/time:output:0.sequential_23/lstm_24/TensorArrayV2_1:handle:0$sequential_23/lstm_24/zeros:output:0&sequential_23/lstm_24/zeros_1:output:0.sequential_23/lstm_24/strided_slice_1:output:0Msequential_23/lstm_24/TensorArrayUnstack/TensorListFromTensor:output_handle:0@sequential_23_lstm_24_lstm_cell_24_split_readvariableop_resourceBsequential_23_lstm_24_lstm_cell_24_split_1_readvariableop_resource:sequential_23_lstm_24_lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *3
body+R)
'sequential_23_lstm_24_while_body_481027*3
cond+R)
'sequential_23_lstm_24_while_cond_481026*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
Fsequential_23/lstm_24/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
8sequential_23/lstm_24/TensorArrayV2Stack/TensorListStackTensorListStack$sequential_23/lstm_24/while:output:3Osequential_23/lstm_24/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0~
+sequential_23/lstm_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿw
-sequential_23/lstm_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-sequential_23/lstm_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:õ
%sequential_23/lstm_24/strided_slice_3StridedSliceAsequential_23/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:04sequential_23/lstm_24/strided_slice_3/stack:output:06sequential_23/lstm_24/strided_slice_3/stack_1:output:06sequential_23/lstm_24/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask{
&sequential_23/lstm_24/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ø
!sequential_23/lstm_24/transpose_1	TransposeAsequential_23/lstm_24/TensorArrayV2Stack/TensorListStack:tensor:0/sequential_23/lstm_24/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
sequential_23/lstm_24/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    ¢
,sequential_23/dense_73/MatMul/ReadVariableOpReadVariableOp5sequential_23_dense_73_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¿
sequential_23/dense_73/MatMulMatMul.sequential_23/lstm_24/strided_slice_3:output:04sequential_23/dense_73/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_23/dense_73/BiasAdd/ReadVariableOpReadVariableOp6sequential_23_dense_73_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_23/dense_73/BiasAddBiasAdd'sequential_23/dense_73/MatMul:product:05sequential_23/dense_73/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
,sequential_23/dense_74/MatMul/ReadVariableOpReadVariableOp5sequential_23_dense_74_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¸
sequential_23/dense_74/MatMulMatMul'sequential_23/dense_73/BiasAdd:output:04sequential_23/dense_74/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
-sequential_23/dense_74/BiasAdd/ReadVariableOpReadVariableOp6sequential_23_dense_74_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0»
sequential_23/dense_74/BiasAddBiasAdd'sequential_23/dense_74/MatMul:product:05sequential_23/dense_74/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_23/dense_74/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÐ
NoOpNoOp.^sequential_23/dense_72/BiasAdd/ReadVariableOp0^sequential_23/dense_72/Tensordot/ReadVariableOp.^sequential_23/dense_73/BiasAdd/ReadVariableOp-^sequential_23/dense_73/MatMul/ReadVariableOp.^sequential_23/dense_74/BiasAdd/ReadVariableOp-^sequential_23/dense_74/MatMul/ReadVariableOp2^sequential_23/lstm_24/lstm_cell_24/ReadVariableOp4^sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_14^sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_24^sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_38^sequential_23/lstm_24/lstm_cell_24/split/ReadVariableOp:^sequential_23/lstm_24/lstm_cell_24/split_1/ReadVariableOp^sequential_23/lstm_24/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2^
-sequential_23/dense_72/BiasAdd/ReadVariableOp-sequential_23/dense_72/BiasAdd/ReadVariableOp2b
/sequential_23/dense_72/Tensordot/ReadVariableOp/sequential_23/dense_72/Tensordot/ReadVariableOp2^
-sequential_23/dense_73/BiasAdd/ReadVariableOp-sequential_23/dense_73/BiasAdd/ReadVariableOp2\
,sequential_23/dense_73/MatMul/ReadVariableOp,sequential_23/dense_73/MatMul/ReadVariableOp2^
-sequential_23/dense_74/BiasAdd/ReadVariableOp-sequential_23/dense_74/BiasAdd/ReadVariableOp2\
,sequential_23/dense_74/MatMul/ReadVariableOp,sequential_23/dense_74/MatMul/ReadVariableOp2f
1sequential_23/lstm_24/lstm_cell_24/ReadVariableOp1sequential_23/lstm_24/lstm_cell_24/ReadVariableOp2j
3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_13sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_12j
3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_23sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_22j
3sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_33sequential_23/lstm_24/lstm_cell_24/ReadVariableOp_32r
7sequential_23/lstm_24/lstm_cell_24/split/ReadVariableOp7sequential_23/lstm_24/lstm_cell_24/split/ReadVariableOp2v
9sequential_23/lstm_24/lstm_cell_24/split_1/ReadVariableOp9sequential_23/lstm_24/lstm_cell_24/split_1/ReadVariableOp2:
sequential_23/lstm_24/whilesequential_23/lstm_24/while:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_72_input
µ
Ã
while_cond_482239
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_482239___redundant_placeholder04
0while_while_cond_482239___redundant_placeholder14
0while_while_cond_482239___redundant_placeholder24
0while_while_cond_482239___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
¸
¶
lstm_24_while_body_482791,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0L
:lstm_24_while_lstm_cell_24_split_readvariableop_resource_0:J
<lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0:F
4lstm_24_while_lstm_cell_24_readvariableop_resource_0:
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorJ
8lstm_24_while_lstm_cell_24_split_readvariableop_resource:H
:lstm_24_while_lstm_cell_24_split_1_readvariableop_resource:D
2lstm_24_while_lstm_cell_24_readvariableop_resource:¢)lstm_24/while/lstm_cell_24/ReadVariableOp¢+lstm_24/while/lstm_cell_24/ReadVariableOp_1¢+lstm_24/while/lstm_cell_24/ReadVariableOp_2¢+lstm_24/while/lstm_cell_24/ReadVariableOp_3¢/lstm_24/while/lstm_cell_24/split/ReadVariableOp¢1lstm_24/while/lstm_cell_24/split_1/ReadVariableOp
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0lstm_24_while_placeholderHlstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
*lstm_24/while/lstm_cell_24/ones_like/ShapeShape8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:o
*lstm_24/while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
$lstm_24/while/lstm_cell_24/ones_likeFill3lstm_24/while/lstm_cell_24/ones_like/Shape:output:03lstm_24/while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,lstm_24/while/lstm_cell_24/ones_like_1/ShapeShapelstm_24_while_placeholder_2*
T0*
_output_shapes
:q
,lstm_24/while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Î
&lstm_24/while/lstm_cell_24/ones_like_1Fill5lstm_24/while/lstm_cell_24/ones_like_1/Shape:output:05lstm_24/while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
lstm_24/while/lstm_cell_24/mulMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
 lstm_24/while/lstm_cell_24/mul_1Mul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
 lstm_24/while/lstm_cell_24/mul_2Mul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÂ
 lstm_24/while/lstm_cell_24/mul_3Mul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ª
/lstm_24/while/lstm_cell_24/split/ReadVariableOpReadVariableOp:lstm_24_while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0ï
 lstm_24/while/lstm_cell_24/splitSplit3lstm_24/while/lstm_cell_24/split/split_dim:output:07lstm_24/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split¬
!lstm_24/while/lstm_cell_24/MatMulMatMul"lstm_24/while/lstm_cell_24/mul:z:0)lstm_24/while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
#lstm_24/while/lstm_cell_24/MatMul_1MatMul$lstm_24/while/lstm_cell_24/mul_1:z:0)lstm_24/while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
#lstm_24/while/lstm_cell_24/MatMul_2MatMul$lstm_24/while/lstm_cell_24/mul_2:z:0)lstm_24/while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
#lstm_24/while/lstm_cell_24/MatMul_3MatMul$lstm_24/while/lstm_cell_24/mul_3:z:0)lstm_24/while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,lstm_24/while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ª
1lstm_24/while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0å
"lstm_24/while/lstm_cell_24/split_1Split5lstm_24/while/lstm_cell_24/split_1/split_dim:output:09lstm_24/while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¹
"lstm_24/while/lstm_cell_24/BiasAddBiasAdd+lstm_24/while/lstm_cell_24/MatMul:product:0+lstm_24/while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
$lstm_24/while/lstm_cell_24/BiasAdd_1BiasAdd-lstm_24/while/lstm_cell_24/MatMul_1:product:0+lstm_24/while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
$lstm_24/while/lstm_cell_24/BiasAdd_2BiasAdd-lstm_24/while/lstm_cell_24/MatMul_2:product:0+lstm_24/while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
$lstm_24/while/lstm_cell_24/BiasAdd_3BiasAdd-lstm_24/while/lstm_cell_24/MatMul_3:product:0+lstm_24/while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
 lstm_24/while/lstm_cell_24/mul_4Mullstm_24_while_placeholder_2/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
 lstm_24/while/lstm_cell_24/mul_5Mullstm_24_while_placeholder_2/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
 lstm_24/while/lstm_cell_24/mul_6Mullstm_24_while_placeholder_2/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
 lstm_24/while/lstm_cell_24/mul_7Mullstm_24_while_placeholder_2/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_24/while/lstm_cell_24/ReadVariableOpReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
.lstm_24/while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0lstm_24/while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_24/while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_24/while/lstm_cell_24/strided_sliceStridedSlice1lstm_24/while/lstm_cell_24/ReadVariableOp:value:07lstm_24/while/lstm_cell_24/strided_slice/stack:output:09lstm_24/while/lstm_cell_24/strided_slice/stack_1:output:09lstm_24/while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¸
#lstm_24/while/lstm_cell_24/MatMul_4MatMul$lstm_24/while/lstm_cell_24/mul_4:z:01lstm_24/while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
lstm_24/while/lstm_cell_24/addAddV2+lstm_24/while/lstm_cell_24/BiasAdd:output:0-lstm_24/while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_24/while/lstm_cell_24/SigmoidSigmoid"lstm_24/while/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+lstm_24/while/lstm_cell_24/ReadVariableOp_1ReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
0lstm_24/while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
*lstm_24/while/lstm_cell_24/strided_slice_1StridedSlice3lstm_24/while/lstm_cell_24/ReadVariableOp_1:value:09lstm_24/while/lstm_cell_24/strided_slice_1/stack:output:0;lstm_24/while/lstm_cell_24/strided_slice_1/stack_1:output:0;lstm_24/while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskº
#lstm_24/while/lstm_cell_24/MatMul_5MatMul$lstm_24/while/lstm_cell_24/mul_5:z:03lstm_24/while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 lstm_24/while/lstm_cell_24/add_1AddV2-lstm_24/while/lstm_cell_24/BiasAdd_1:output:0-lstm_24/while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid$lstm_24/while/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_24/while/lstm_cell_24/mul_8Mul(lstm_24/while/lstm_cell_24/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+lstm_24/while/lstm_cell_24/ReadVariableOp_2ReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
0lstm_24/while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
*lstm_24/while/lstm_cell_24/strided_slice_2StridedSlice3lstm_24/while/lstm_cell_24/ReadVariableOp_2:value:09lstm_24/while/lstm_cell_24/strided_slice_2/stack:output:0;lstm_24/while/lstm_cell_24/strided_slice_2/stack_1:output:0;lstm_24/while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskº
#lstm_24/while/lstm_cell_24/MatMul_6MatMul$lstm_24/while/lstm_cell_24/mul_6:z:03lstm_24/while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 lstm_24/while/lstm_cell_24/add_2AddV2-lstm_24/while/lstm_cell_24/BiasAdd_2:output:0-lstm_24/while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid$lstm_24/while/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
 lstm_24/while/lstm_cell_24/mul_9Mul&lstm_24/while/lstm_cell_24/Sigmoid:y:0(lstm_24/while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
 lstm_24/while/lstm_cell_24/add_3AddV2$lstm_24/while/lstm_cell_24/mul_8:z:0$lstm_24/while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+lstm_24/while/lstm_cell_24/ReadVariableOp_3ReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
0lstm_24/while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
2lstm_24/while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
*lstm_24/while/lstm_cell_24/strided_slice_3StridedSlice3lstm_24/while/lstm_cell_24/ReadVariableOp_3:value:09lstm_24/while/lstm_cell_24/strided_slice_3/stack:output:0;lstm_24/while/lstm_cell_24/strided_slice_3/stack_1:output:0;lstm_24/while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskº
#lstm_24/while/lstm_cell_24/MatMul_7MatMul$lstm_24/while/lstm_cell_24/mul_7:z:03lstm_24/while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 lstm_24/while/lstm_cell_24/add_4AddV2-lstm_24/while/lstm_cell_24/BiasAdd_3:output:0-lstm_24/while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_3Sigmoid$lstm_24/while/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_4Sigmoid$lstm_24/while/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
!lstm_24/while/lstm_cell_24/mul_10Mul(lstm_24/while/lstm_cell_24/Sigmoid_3:y:0(lstm_24/while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
2lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_24_while_placeholder_1lstm_24_while_placeholder%lstm_24/while/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_24/while/addAddV2lstm_24_while_placeholderlstm_24/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_24/while/add_1AddV2(lstm_24_while_lstm_24_while_loop_counterlstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_24/while/IdentityIdentitylstm_24/while/add_1:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 
lstm_24/while/Identity_1Identity.lstm_24_while_lstm_24_while_maximum_iterations^lstm_24/while/NoOp*
T0*
_output_shapes
: q
lstm_24/while/Identity_2Identitylstm_24/while/add:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: ±
lstm_24/while/Identity_3IdentityBlstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_24/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_24/while/Identity_4Identity%lstm_24/while/lstm_cell_24/mul_10:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_24/add_3:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_24/while/NoOpNoOp*^lstm_24/while/lstm_cell_24/ReadVariableOp,^lstm_24/while/lstm_cell_24/ReadVariableOp_1,^lstm_24/while/lstm_cell_24/ReadVariableOp_2,^lstm_24/while/lstm_cell_24/ReadVariableOp_30^lstm_24/while/lstm_cell_24/split/ReadVariableOp2^lstm_24/while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_24_while_identitylstm_24/while/Identity:output:0"=
lstm_24_while_identity_1!lstm_24/while/Identity_1:output:0"=
lstm_24_while_identity_2!lstm_24/while/Identity_2:output:0"=
lstm_24_while_identity_3!lstm_24/while/Identity_3:output:0"=
lstm_24_while_identity_4!lstm_24/while/Identity_4:output:0"=
lstm_24_while_identity_5!lstm_24/while/Identity_5:output:0"P
%lstm_24_while_lstm_24_strided_slice_1'lstm_24_while_lstm_24_strided_slice_1_0"j
2lstm_24_while_lstm_cell_24_readvariableop_resource4lstm_24_while_lstm_cell_24_readvariableop_resource_0"z
:lstm_24_while_lstm_cell_24_split_1_readvariableop_resource<lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0"v
8lstm_24_while_lstm_cell_24_split_readvariableop_resource:lstm_24_while_lstm_cell_24_split_readvariableop_resource_0"È
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)lstm_24/while/lstm_cell_24/ReadVariableOp)lstm_24/while/lstm_cell_24/ReadVariableOp2Z
+lstm_24/while/lstm_cell_24/ReadVariableOp_1+lstm_24/while/lstm_cell_24/ReadVariableOp_12Z
+lstm_24/while/lstm_cell_24/ReadVariableOp_2+lstm_24/while/lstm_cell_24/ReadVariableOp_22Z
+lstm_24/while/lstm_cell_24/ReadVariableOp_3+lstm_24/while/lstm_cell_24/ReadVariableOp_32b
/lstm_24/while/lstm_cell_24/split/ReadVariableOp/lstm_24/while/lstm_cell_24/split/ReadVariableOp2f
1lstm_24/while/lstm_cell_24/split_1/ReadVariableOp1lstm_24/while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ç	
õ
D__inference_dense_74_layer_call_and_return_conditional_losses_484720

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
v
	
while_body_483563
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_24_split_readvariableop_resource_0:B
4while_lstm_cell_24_split_1_readvariableop_resource_0:>
,while_lstm_cell_24_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_24_split_readvariableop_resource:@
2while_lstm_cell_24_split_1_readvariableop_resource:<
*while_lstm_cell_24_readvariableop_resource:¢!while/lstm_cell_24/ReadVariableOp¢#while/lstm_cell_24/ReadVariableOp_1¢#while/lstm_cell_24/ReadVariableOp_2¢#while/lstm_cell_24/ReadVariableOp_3¢'while/lstm_cell_24/split/ReadVariableOp¢)while/lstm_cell_24/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
"while/lstm_cell_24/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
while/lstm_cell_24/ones_likeFill+while/lstm_cell_24/ones_like/Shape:output:0+while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$while/lstm_cell_24/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
while/lstm_cell_24/ones_like_1Fill-while/lstm_cell_24/ones_like_1/Shape:output:0-while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
while/lstm_cell_24/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_24/split/ReadVariableOpReadVariableOp2while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0×
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
while/lstm_cell_24/MatMulMatMulwhile/lstm_cell_24/mul:z:0!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_1MatMulwhile/lstm_cell_24/mul_1:z:0!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_2MatMulwhile/lstm_cell_24/mul_2:z:0!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_3MatMulwhile/lstm_cell_24/mul_3:z:0!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0Í
while/lstm_cell_24/split_1Split-while/lstm_cell_24/split_1/split_dim:output:01while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¡
while/lstm_cell_24/BiasAddBiasAdd#while/lstm_cell_24/MatMul:product:0#while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_1BiasAdd%while/lstm_cell_24/MatMul_1:product:0#while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_2BiasAdd%while/lstm_cell_24/MatMul_2:product:0#while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_3BiasAdd%while/lstm_cell_24/MatMul_3:product:0#while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_4Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_5Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_6Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_7Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/ReadVariableOpReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 while/lstm_cell_24/strided_sliceStridedSlice)while/lstm_cell_24/ReadVariableOp:value:0/while/lstm_cell_24/strided_slice/stack:output:01while/lstm_cell_24/strided_slice/stack_1:output:01while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask 
while/lstm_cell_24/MatMul_4MatMulwhile/lstm_cell_24/mul_4:z:0)while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/addAddV2#while/lstm_cell_24/BiasAdd:output:0%while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_24/SigmoidSigmoidwhile/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_1ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_1StridedSlice+while/lstm_cell_24/ReadVariableOp_1:value:01while/lstm_cell_24/strided_slice_1/stack:output:03while/lstm_cell_24/strided_slice_1/stack_1:output:03while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_5MatMulwhile/lstm_cell_24/mul_5:z:0+while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_1AddV2%while/lstm_cell_24/BiasAdd_1:output:0%while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_1Sigmoidwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_8Mul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_2ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_2StridedSlice+while/lstm_cell_24/ReadVariableOp_2:value:01while/lstm_cell_24/strided_slice_2/stack:output:03while/lstm_cell_24/strided_slice_2/stack_1:output:03while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_6MatMulwhile/lstm_cell_24/mul_6:z:0+while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_2AddV2%while/lstm_cell_24/BiasAdd_2:output:0%while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_2Sigmoidwhile/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_9Mulwhile/lstm_cell_24/Sigmoid:y:0 while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/add_3AddV2while/lstm_cell_24/mul_8:z:0while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_3ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_3StridedSlice+while/lstm_cell_24/ReadVariableOp_3:value:01while/lstm_cell_24/strided_slice_3/stack:output:03while/lstm_cell_24/strided_slice_3/stack_1:output:03while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_7MatMulwhile/lstm_cell_24/mul_7:z:0+while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_4AddV2%while/lstm_cell_24/BiasAdd_3:output:0%while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_3Sigmoidwhile/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_4Sigmoidwhile/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_10Mul while/lstm_cell_24/Sigmoid_3:y:0 while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_24/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_24/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸

while/NoOpNoOp"^while/lstm_cell_24/ReadVariableOp$^while/lstm_cell_24/ReadVariableOp_1$^while/lstm_cell_24/ReadVariableOp_2$^while/lstm_cell_24/ReadVariableOp_3(^while/lstm_cell_24/split/ReadVariableOp*^while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_24_readvariableop_resource,while_lstm_cell_24_readvariableop_resource_0"j
2while_lstm_cell_24_split_1_readvariableop_resource4while_lstm_cell_24_split_1_readvariableop_resource_0"f
0while_lstm_cell_24_split_readvariableop_resource2while_lstm_cell_24_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2F
!while/lstm_cell_24/ReadVariableOp!while/lstm_cell_24/ReadVariableOp2J
#while/lstm_cell_24/ReadVariableOp_1#while/lstm_cell_24/ReadVariableOp_12J
#while/lstm_cell_24/ReadVariableOp_2#while/lstm_cell_24/ReadVariableOp_22J
#while/lstm_cell_24/ReadVariableOp_3#while/lstm_cell_24/ReadVariableOp_32R
'while/lstm_cell_24/split/ReadVariableOp'while/lstm_cell_24/split/ReadVariableOp2V
)while/lstm_cell_24/split_1/ReadVariableOp)while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Ç	
õ
D__inference_dense_74_layer_call_and_return_conditional_losses_482006

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

æ
C__inference_lstm_24_layer_call_and_return_conditional_losses_483697
inputs_0<
*lstm_cell_24_split_readvariableop_resource::
,lstm_cell_24_split_1_readvariableop_resource:6
$lstm_cell_24_readvariableop_resource:
identity¢lstm_cell_24/ReadVariableOp¢lstm_cell_24/ReadVariableOp_1¢lstm_cell_24/ReadVariableOp_2¢lstm_cell_24/ReadVariableOp_3¢!lstm_cell_24/split/ReadVariableOp¢#lstm_cell_24/split_1/ReadVariableOp¢while=
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskd
lstm_cell_24/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_24/ones_likeFill%lstm_cell_24/ones_like/Shape:output:0%lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
lstm_cell_24/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm_cell_24/ones_like_1Fill'lstm_cell_24/ones_like_1/Shape:output:0'lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mulMulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_1Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_2Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_3Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_24/split/ReadVariableOpReadVariableOp*lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Å
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0)lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_cell_24/MatMulMatMullstm_cell_24/mul:z:0lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_1MatMullstm_cell_24/mul_1:z:0lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_2MatMullstm_cell_24/mul_2:z:0lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_3MatMullstm_cell_24/mul_3:z:0lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_24/split_1/ReadVariableOpReadVariableOp,lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0»
lstm_cell_24/split_1Split'lstm_cell_24/split_1/split_dim:output:0+lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
lstm_cell_24/BiasAddBiasAddlstm_cell_24/MatMul:product:0lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_1BiasAddlstm_cell_24/MatMul_1:product:0lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_2BiasAddlstm_cell_24/MatMul_2:product:0lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_3BiasAddlstm_cell_24/MatMul_3:product:0lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_4Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_5Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_6Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_7Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOpReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¬
lstm_cell_24/strided_sliceStridedSlice#lstm_cell_24/ReadVariableOp:value:0)lstm_cell_24/strided_slice/stack:output:0+lstm_cell_24/strided_slice/stack_1:output:0+lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_4MatMullstm_cell_24/mul_4:z:0#lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/addAddV2lstm_cell_24/BiasAdd:output:0lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_24/SigmoidSigmoidlstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_1ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_1StridedSlice%lstm_cell_24/ReadVariableOp_1:value:0+lstm_cell_24/strided_slice_1/stack:output:0-lstm_cell_24/strided_slice_1/stack_1:output:0-lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_5MatMullstm_cell_24/mul_5:z:0%lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_1AddV2lstm_cell_24/BiasAdd_1:output:0lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_24/mul_8Mullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_2ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_2StridedSlice%lstm_cell_24/ReadVariableOp_2:value:0+lstm_cell_24/strided_slice_2/stack:output:0-lstm_cell_24/strided_slice_2/stack_1:output:0-lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_6MatMullstm_cell_24/mul_6:z:0%lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_2AddV2lstm_cell_24/BiasAdd_2:output:0lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_9Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/add_3AddV2lstm_cell_24/mul_8:z:0lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_3ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_3StridedSlice%lstm_cell_24/ReadVariableOp_3:value:0+lstm_cell_24/strided_slice_3/stack:output:0-lstm_cell_24/strided_slice_3/stack_1:output:0-lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_7MatMullstm_cell_24/mul_7:z:0%lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_4AddV2lstm_cell_24/BiasAdd_3:output:0lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_3Sigmoidlstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_4Sigmoidlstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_10Mullstm_cell_24/Sigmoid_3:y:0lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_24_split_readvariableop_resource,lstm_cell_24_split_1_readvariableop_resource$lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_483563*
condR
while_cond_483562*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell_24/ReadVariableOp^lstm_cell_24/ReadVariableOp_1^lstm_cell_24/ReadVariableOp_2^lstm_cell_24/ReadVariableOp_3"^lstm_cell_24/split/ReadVariableOp$^lstm_cell_24/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2:
lstm_cell_24/ReadVariableOplstm_cell_24/ReadVariableOp2>
lstm_cell_24/ReadVariableOp_1lstm_cell_24/ReadVariableOp_12>
lstm_cell_24/ReadVariableOp_2lstm_cell_24/ReadVariableOp_22>
lstm_cell_24/ReadVariableOp_3lstm_cell_24/ReadVariableOp_32F
!lstm_cell_24/split/ReadVariableOp!lstm_cell_24/split/ReadVariableOp2J
#lstm_cell_24/split_1/ReadVariableOp#lstm_cell_24/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
D
¦
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481290

inputs

states
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_4Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_5Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_6Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
mul_7Mulstatesones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
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
valueB"      ë
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
:ÿÿÿÿÿÿÿÿÿd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates


ã
lstm_24_while_cond_483135,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1D
@lstm_24_while_lstm_24_while_cond_483135___redundant_placeholder0D
@lstm_24_while_lstm_24_while_cond_483135___redundant_placeholder1D
@lstm_24_while_lstm_24_while_cond_483135___redundant_placeholder2D
@lstm_24_while_lstm_24_while_cond_483135___redundant_placeholder3
lstm_24_while_identity

lstm_24/while/LessLesslstm_24_while_placeholder*lstm_24_while_less_lstm_24_strided_slice_1*
T0*
_output_shapes
: [
lstm_24/while/IdentityIdentitylstm_24/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_24_while_identitylstm_24/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_481608
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_481608___redundant_placeholder04
0while_while_cond_481608___redundant_placeholder14
0while_while_cond_481608___redundant_placeholder24
0while_while_cond_481608___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
Ò

)__inference_dense_72_layer_call_fn_483380

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_481724s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
ä
C__inference_lstm_24_layer_call_and_return_conditional_losses_484311

inputs<
*lstm_cell_24_split_readvariableop_resource::
,lstm_cell_24_split_1_readvariableop_resource:6
$lstm_cell_24_readvariableop_resource:
identity¢lstm_cell_24/ReadVariableOp¢lstm_cell_24/ReadVariableOp_1¢lstm_cell_24/ReadVariableOp_2¢lstm_cell_24/ReadVariableOp_3¢!lstm_cell_24/split/ReadVariableOp¢#lstm_cell_24/split_1/ReadVariableOp¢while;
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskd
lstm_cell_24/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_24/ones_likeFill%lstm_cell_24/ones_like/Shape:output:0%lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
lstm_cell_24/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm_cell_24/ones_like_1Fill'lstm_cell_24/ones_like_1/Shape:output:0'lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mulMulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_1Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_2Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_3Mulstrided_slice_2:output:0lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_24/split/ReadVariableOpReadVariableOp*lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Å
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0)lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_cell_24/MatMulMatMullstm_cell_24/mul:z:0lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_1MatMullstm_cell_24/mul_1:z:0lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_2MatMullstm_cell_24/mul_2:z:0lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_3MatMullstm_cell_24/mul_3:z:0lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_24/split_1/ReadVariableOpReadVariableOp,lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0»
lstm_cell_24/split_1Split'lstm_cell_24/split_1/split_dim:output:0+lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
lstm_cell_24/BiasAddBiasAddlstm_cell_24/MatMul:product:0lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_1BiasAddlstm_cell_24/MatMul_1:product:0lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_2BiasAddlstm_cell_24/MatMul_2:product:0lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_3BiasAddlstm_cell_24/MatMul_3:product:0lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_4Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_5Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_6Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ~
lstm_cell_24/mul_7Mulzeros:output:0!lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOpReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¬
lstm_cell_24/strided_sliceStridedSlice#lstm_cell_24/ReadVariableOp:value:0)lstm_cell_24/strided_slice/stack:output:0+lstm_cell_24/strided_slice/stack_1:output:0+lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_4MatMullstm_cell_24/mul_4:z:0#lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/addAddV2lstm_cell_24/BiasAdd:output:0lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_24/SigmoidSigmoidlstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_1ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_1StridedSlice%lstm_cell_24/ReadVariableOp_1:value:0+lstm_cell_24/strided_slice_1/stack:output:0-lstm_cell_24/strided_slice_1/stack_1:output:0-lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_5MatMullstm_cell_24/mul_5:z:0%lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_1AddV2lstm_cell_24/BiasAdd_1:output:0lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_24/mul_8Mullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_2ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_2StridedSlice%lstm_cell_24/ReadVariableOp_2:value:0+lstm_cell_24/strided_slice_2/stack:output:0-lstm_cell_24/strided_slice_2/stack_1:output:0-lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_6MatMullstm_cell_24/mul_6:z:0%lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_2AddV2lstm_cell_24/BiasAdd_2:output:0lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_9Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/add_3AddV2lstm_cell_24/mul_8:z:0lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_3ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_3StridedSlice%lstm_cell_24/ReadVariableOp_3:value:0+lstm_cell_24/strided_slice_3/stack:output:0-lstm_cell_24/strided_slice_3/stack_1:output:0-lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_7MatMullstm_cell_24/mul_7:z:0%lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_4AddV2lstm_cell_24/BiasAdd_3:output:0lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_3Sigmoidlstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_4Sigmoidlstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_10Mullstm_cell_24/Sigmoid_3:y:0lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_24_split_readvariableop_resource,lstm_cell_24_split_1_readvariableop_resource$lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_484177*
condR
while_cond_484176*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell_24/ReadVariableOp^lstm_cell_24/ReadVariableOp_1^lstm_cell_24/ReadVariableOp_2^lstm_cell_24/ReadVariableOp_3"^lstm_cell_24/split/ReadVariableOp$^lstm_cell_24/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2:
lstm_cell_24/ReadVariableOplstm_cell_24/ReadVariableOp2>
lstm_cell_24/ReadVariableOp_1lstm_cell_24/ReadVariableOp_12>
lstm_cell_24/ReadVariableOp_2lstm_cell_24/ReadVariableOp_22>
lstm_cell_24/ReadVariableOp_3lstm_cell_24/ReadVariableOp_32F
!lstm_cell_24/split/ReadVariableOp!lstm_cell_24/split/ReadVariableOp2J
#lstm_cell_24/split_1/ReadVariableOp#lstm_cell_24/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
Ú
.__inference_sequential_23_layer_call_fn_482656

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_482508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÐÞ
¶
lstm_24_while_body_483136,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3+
'lstm_24_while_lstm_24_strided_slice_1_0g
clstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0L
:lstm_24_while_lstm_cell_24_split_readvariableop_resource_0:J
<lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0:F
4lstm_24_while_lstm_cell_24_readvariableop_resource_0:
lstm_24_while_identity
lstm_24_while_identity_1
lstm_24_while_identity_2
lstm_24_while_identity_3
lstm_24_while_identity_4
lstm_24_while_identity_5)
%lstm_24_while_lstm_24_strided_slice_1e
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorJ
8lstm_24_while_lstm_cell_24_split_readvariableop_resource:H
:lstm_24_while_lstm_cell_24_split_1_readvariableop_resource:D
2lstm_24_while_lstm_cell_24_readvariableop_resource:¢)lstm_24/while/lstm_cell_24/ReadVariableOp¢+lstm_24/while/lstm_cell_24/ReadVariableOp_1¢+lstm_24/while/lstm_cell_24/ReadVariableOp_2¢+lstm_24/while/lstm_cell_24/ReadVariableOp_3¢/lstm_24/while/lstm_cell_24/split/ReadVariableOp¢1lstm_24/while/lstm_cell_24/split_1/ReadVariableOp
?lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Î
1lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0lstm_24_while_placeholderHlstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
*lstm_24/while/lstm_cell_24/ones_like/ShapeShape8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:o
*lstm_24/while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?È
$lstm_24/while/lstm_cell_24/ones_likeFill3lstm_24/while/lstm_cell_24/ones_like/Shape:output:03lstm_24/while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
(lstm_24/while/lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Á
&lstm_24/while/lstm_cell_24/dropout/MulMul-lstm_24/while/lstm_cell_24/ones_like:output:01lstm_24/while/lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(lstm_24/while/lstm_cell_24/dropout/ShapeShape-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:Â
?lstm_24/while/lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform1lstm_24/while/lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0v
1lstm_24/while/lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=÷
/lstm_24/while/lstm_cell_24/dropout/GreaterEqualGreaterEqualHlstm_24/while/lstm_cell_24/dropout/random_uniform/RandomUniform:output:0:lstm_24/while/lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
'lstm_24/while/lstm_cell_24/dropout/CastCast3lstm_24/while/lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
(lstm_24/while/lstm_cell_24/dropout/Mul_1Mul*lstm_24/while/lstm_cell_24/dropout/Mul:z:0+lstm_24/while/lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*lstm_24/while/lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Å
(lstm_24/while/lstm_cell_24/dropout_1/MulMul-lstm_24/while/lstm_cell_24/ones_like:output:03lstm_24/while/lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_24/while/lstm_cell_24/dropout_1/ShapeShape-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:Æ
Alstm_24/while/lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform3lstm_24/while/lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0x
3lstm_24/while/lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ý
1lstm_24/while/lstm_cell_24/dropout_1/GreaterEqualGreaterEqualJlstm_24/while/lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:0<lstm_24/while/lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
)lstm_24/while/lstm_cell_24/dropout_1/CastCast5lstm_24/while/lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
*lstm_24/while/lstm_cell_24/dropout_1/Mul_1Mul,lstm_24/while/lstm_cell_24/dropout_1/Mul:z:0-lstm_24/while/lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*lstm_24/while/lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Å
(lstm_24/while/lstm_cell_24/dropout_2/MulMul-lstm_24/while/lstm_cell_24/ones_like:output:03lstm_24/while/lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_24/while/lstm_cell_24/dropout_2/ShapeShape-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:Æ
Alstm_24/while/lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform3lstm_24/while/lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0x
3lstm_24/while/lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ý
1lstm_24/while/lstm_cell_24/dropout_2/GreaterEqualGreaterEqualJlstm_24/while/lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:0<lstm_24/while/lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
)lstm_24/while/lstm_cell_24/dropout_2/CastCast5lstm_24/while/lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
*lstm_24/while/lstm_cell_24/dropout_2/Mul_1Mul,lstm_24/while/lstm_cell_24/dropout_2/Mul:z:0-lstm_24/while/lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*lstm_24/while/lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Å
(lstm_24/while/lstm_cell_24/dropout_3/MulMul-lstm_24/while/lstm_cell_24/ones_like:output:03lstm_24/while/lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_24/while/lstm_cell_24/dropout_3/ShapeShape-lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:Æ
Alstm_24/while/lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform3lstm_24/while/lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0x
3lstm_24/while/lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ý
1lstm_24/while/lstm_cell_24/dropout_3/GreaterEqualGreaterEqualJlstm_24/while/lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:0<lstm_24/while/lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
)lstm_24/while/lstm_cell_24/dropout_3/CastCast5lstm_24/while/lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
*lstm_24/while/lstm_cell_24/dropout_3/Mul_1Mul,lstm_24/while/lstm_cell_24/dropout_3/Mul:z:0-lstm_24/while/lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
,lstm_24/while/lstm_cell_24/ones_like_1/ShapeShapelstm_24_while_placeholder_2*
T0*
_output_shapes
:q
,lstm_24/while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?Î
&lstm_24/while/lstm_cell_24/ones_like_1Fill5lstm_24/while/lstm_cell_24/ones_like_1/Shape:output:05lstm_24/while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*lstm_24/while/lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Ç
(lstm_24/while/lstm_cell_24/dropout_4/MulMul/lstm_24/while/lstm_cell_24/ones_like_1:output:03lstm_24/while/lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_24/while/lstm_cell_24/dropout_4/ShapeShape/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:Æ
Alstm_24/while/lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform3lstm_24/while/lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0x
3lstm_24/while/lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ý
1lstm_24/while/lstm_cell_24/dropout_4/GreaterEqualGreaterEqualJlstm_24/while/lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:0<lstm_24/while/lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
)lstm_24/while/lstm_cell_24/dropout_4/CastCast5lstm_24/while/lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
*lstm_24/while/lstm_cell_24/dropout_4/Mul_1Mul,lstm_24/while/lstm_cell_24/dropout_4/Mul:z:0-lstm_24/while/lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*lstm_24/while/lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Ç
(lstm_24/while/lstm_cell_24/dropout_5/MulMul/lstm_24/while/lstm_cell_24/ones_like_1:output:03lstm_24/while/lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_24/while/lstm_cell_24/dropout_5/ShapeShape/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:Æ
Alstm_24/while/lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform3lstm_24/while/lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0x
3lstm_24/while/lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ý
1lstm_24/while/lstm_cell_24/dropout_5/GreaterEqualGreaterEqualJlstm_24/while/lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:0<lstm_24/while/lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
)lstm_24/while/lstm_cell_24/dropout_5/CastCast5lstm_24/while/lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
*lstm_24/while/lstm_cell_24/dropout_5/Mul_1Mul,lstm_24/while/lstm_cell_24/dropout_5/Mul:z:0-lstm_24/while/lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*lstm_24/while/lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Ç
(lstm_24/while/lstm_cell_24/dropout_6/MulMul/lstm_24/while/lstm_cell_24/ones_like_1:output:03lstm_24/while/lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_24/while/lstm_cell_24/dropout_6/ShapeShape/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:Æ
Alstm_24/while/lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform3lstm_24/while/lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0x
3lstm_24/while/lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ý
1lstm_24/while/lstm_cell_24/dropout_6/GreaterEqualGreaterEqualJlstm_24/while/lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:0<lstm_24/while/lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
)lstm_24/while/lstm_cell_24/dropout_6/CastCast5lstm_24/while/lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
*lstm_24/while/lstm_cell_24/dropout_6/Mul_1Mul,lstm_24/while/lstm_cell_24/dropout_6/Mul:z:0-lstm_24/while/lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
*lstm_24/while/lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?Ç
(lstm_24/while/lstm_cell_24/dropout_7/MulMul/lstm_24/while/lstm_cell_24/ones_like_1:output:03lstm_24/while/lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_24/while/lstm_cell_24/dropout_7/ShapeShape/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:Æ
Alstm_24/while/lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform3lstm_24/while/lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0x
3lstm_24/while/lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=ý
1lstm_24/while/lstm_cell_24/dropout_7/GreaterEqualGreaterEqualJlstm_24/while/lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:0<lstm_24/while/lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ©
)lstm_24/while/lstm_cell_24/dropout_7/CastCast5lstm_24/while/lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
*lstm_24/while/lstm_cell_24/dropout_7/Mul_1Mul,lstm_24/while/lstm_cell_24/dropout_7/Mul:z:0-lstm_24/while/lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
lstm_24/while/lstm_cell_24/mulMul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0,lstm_24/while/lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
 lstm_24/while/lstm_cell_24/mul_1Mul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm_24/while/lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
 lstm_24/while/lstm_cell_24/mul_2Mul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm_24/while/lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÃ
 lstm_24/while/lstm_cell_24/mul_3Mul8lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0.lstm_24/while/lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
*lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ª
/lstm_24/while/lstm_cell_24/split/ReadVariableOpReadVariableOp:lstm_24_while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0ï
 lstm_24/while/lstm_cell_24/splitSplit3lstm_24/while/lstm_cell_24/split/split_dim:output:07lstm_24/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split¬
!lstm_24/while/lstm_cell_24/MatMulMatMul"lstm_24/while/lstm_cell_24/mul:z:0)lstm_24/while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
#lstm_24/while/lstm_cell_24/MatMul_1MatMul$lstm_24/while/lstm_cell_24/mul_1:z:0)lstm_24/while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
#lstm_24/while/lstm_cell_24/MatMul_2MatMul$lstm_24/while/lstm_cell_24/mul_2:z:0)lstm_24/while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
#lstm_24/while/lstm_cell_24/MatMul_3MatMul$lstm_24/while/lstm_cell_24/mul_3:z:0)lstm_24/while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
,lstm_24/while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : ª
1lstm_24/while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp<lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0å
"lstm_24/while/lstm_cell_24/split_1Split5lstm_24/while/lstm_cell_24/split_1/split_dim:output:09lstm_24/while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¹
"lstm_24/while/lstm_cell_24/BiasAddBiasAdd+lstm_24/while/lstm_cell_24/MatMul:product:0+lstm_24/while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
$lstm_24/while/lstm_cell_24/BiasAdd_1BiasAdd-lstm_24/while/lstm_cell_24/MatMul_1:product:0+lstm_24/while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
$lstm_24/while/lstm_cell_24/BiasAdd_2BiasAdd-lstm_24/while/lstm_cell_24/MatMul_2:product:0+lstm_24/while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ½
$lstm_24/while/lstm_cell_24/BiasAdd_3BiasAdd-lstm_24/while/lstm_cell_24/MatMul_3:product:0+lstm_24/while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_24/while/lstm_cell_24/mul_4Mullstm_24_while_placeholder_2.lstm_24/while/lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_24/while/lstm_cell_24/mul_5Mullstm_24_while_placeholder_2.lstm_24/while/lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_24/while/lstm_cell_24/mul_6Mullstm_24_while_placeholder_2.lstm_24/while/lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
 lstm_24/while/lstm_cell_24/mul_7Mullstm_24_while_placeholder_2.lstm_24/while/lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_24/while/lstm_cell_24/ReadVariableOpReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
.lstm_24/while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
0lstm_24/while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
0lstm_24/while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ò
(lstm_24/while/lstm_cell_24/strided_sliceStridedSlice1lstm_24/while/lstm_cell_24/ReadVariableOp:value:07lstm_24/while/lstm_cell_24/strided_slice/stack:output:09lstm_24/while/lstm_cell_24/strided_slice/stack_1:output:09lstm_24/while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¸
#lstm_24/while/lstm_cell_24/MatMul_4MatMul$lstm_24/while/lstm_cell_24/mul_4:z:01lstm_24/while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿµ
lstm_24/while/lstm_cell_24/addAddV2+lstm_24/while/lstm_cell_24/BiasAdd:output:0-lstm_24/while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_24/while/lstm_cell_24/SigmoidSigmoid"lstm_24/while/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+lstm_24/while/lstm_cell_24/ReadVariableOp_1ReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
0lstm_24/while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
*lstm_24/while/lstm_cell_24/strided_slice_1StridedSlice3lstm_24/while/lstm_cell_24/ReadVariableOp_1:value:09lstm_24/while/lstm_cell_24/strided_slice_1/stack:output:0;lstm_24/while/lstm_cell_24/strided_slice_1/stack_1:output:0;lstm_24/while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskº
#lstm_24/while/lstm_cell_24/MatMul_5MatMul$lstm_24/while/lstm_cell_24/mul_5:z:03lstm_24/while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 lstm_24/while/lstm_cell_24/add_1AddV2-lstm_24/while/lstm_cell_24/BiasAdd_1:output:0-lstm_24/while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid$lstm_24/while/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
 lstm_24/while/lstm_cell_24/mul_8Mul(lstm_24/while/lstm_cell_24/Sigmoid_1:y:0lstm_24_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+lstm_24/while/lstm_cell_24/ReadVariableOp_2ReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
0lstm_24/while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
*lstm_24/while/lstm_cell_24/strided_slice_2StridedSlice3lstm_24/while/lstm_cell_24/ReadVariableOp_2:value:09lstm_24/while/lstm_cell_24/strided_slice_2/stack:output:0;lstm_24/while/lstm_cell_24/strided_slice_2/stack_1:output:0;lstm_24/while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskº
#lstm_24/while/lstm_cell_24/MatMul_6MatMul$lstm_24/while/lstm_cell_24/mul_6:z:03lstm_24/while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 lstm_24/while/lstm_cell_24/add_2AddV2-lstm_24/while/lstm_cell_24/BiasAdd_2:output:0-lstm_24/while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid$lstm_24/while/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ«
 lstm_24/while/lstm_cell_24/mul_9Mul&lstm_24/while/lstm_cell_24/Sigmoid:y:0(lstm_24/while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
 lstm_24/while/lstm_cell_24/add_3AddV2$lstm_24/while/lstm_cell_24/mul_8:z:0$lstm_24/while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ 
+lstm_24/while/lstm_cell_24/ReadVariableOp_3ReadVariableOp4lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
0lstm_24/while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
2lstm_24/while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
2lstm_24/while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ü
*lstm_24/while/lstm_cell_24/strided_slice_3StridedSlice3lstm_24/while/lstm_cell_24/ReadVariableOp_3:value:09lstm_24/while/lstm_cell_24/strided_slice_3/stack:output:0;lstm_24/while/lstm_cell_24/strided_slice_3/stack_1:output:0;lstm_24/while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskº
#lstm_24/while/lstm_cell_24/MatMul_7MatMul$lstm_24/while/lstm_cell_24/mul_7:z:03lstm_24/while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
 lstm_24/while/lstm_cell_24/add_4AddV2-lstm_24/while/lstm_cell_24/BiasAdd_3:output:0-lstm_24/while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_3Sigmoid$lstm_24/while/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$lstm_24/while/lstm_cell_24/Sigmoid_4Sigmoid$lstm_24/while/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ®
!lstm_24/while/lstm_cell_24/mul_10Mul(lstm_24/while/lstm_cell_24/Sigmoid_3:y:0(lstm_24/while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿæ
2lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_24_while_placeholder_1lstm_24_while_placeholder%lstm_24/while/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒU
lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :t
lstm_24/while/addAddV2lstm_24_while_placeholderlstm_24/while/add/y:output:0*
T0*
_output_shapes
: W
lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_24/while/add_1AddV2(lstm_24_while_lstm_24_while_loop_counterlstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: q
lstm_24/while/IdentityIdentitylstm_24/while/add_1:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: 
lstm_24/while/Identity_1Identity.lstm_24_while_lstm_24_while_maximum_iterations^lstm_24/while/NoOp*
T0*
_output_shapes
: q
lstm_24/while/Identity_2Identitylstm_24/while/add:z:0^lstm_24/while/NoOp*
T0*
_output_shapes
: ±
lstm_24/while/Identity_3IdentityBlstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_24/while/NoOp*
T0*
_output_shapes
: :éèÒ
lstm_24/while/Identity_4Identity%lstm_24/while/lstm_cell_24/mul_10:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_24/while/Identity_5Identity$lstm_24/while/lstm_cell_24/add_3:z:0^lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿð
lstm_24/while/NoOpNoOp*^lstm_24/while/lstm_cell_24/ReadVariableOp,^lstm_24/while/lstm_cell_24/ReadVariableOp_1,^lstm_24/while/lstm_cell_24/ReadVariableOp_2,^lstm_24/while/lstm_cell_24/ReadVariableOp_30^lstm_24/while/lstm_cell_24/split/ReadVariableOp2^lstm_24/while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "9
lstm_24_while_identitylstm_24/while/Identity:output:0"=
lstm_24_while_identity_1!lstm_24/while/Identity_1:output:0"=
lstm_24_while_identity_2!lstm_24/while/Identity_2:output:0"=
lstm_24_while_identity_3!lstm_24/while/Identity_3:output:0"=
lstm_24_while_identity_4!lstm_24/while/Identity_4:output:0"=
lstm_24_while_identity_5!lstm_24/while/Identity_5:output:0"P
%lstm_24_while_lstm_24_strided_slice_1'lstm_24_while_lstm_24_strided_slice_1_0"j
2lstm_24_while_lstm_cell_24_readvariableop_resource4lstm_24_while_lstm_cell_24_readvariableop_resource_0"z
:lstm_24_while_lstm_cell_24_split_1_readvariableop_resource<lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0"v
8lstm_24_while_lstm_cell_24_split_readvariableop_resource:lstm_24_while_lstm_cell_24_split_readvariableop_resource_0"È
alstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensorclstm_24_while_tensorarrayv2read_tensorlistgetitem_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2V
)lstm_24/while/lstm_cell_24/ReadVariableOp)lstm_24/while/lstm_cell_24/ReadVariableOp2Z
+lstm_24/while/lstm_cell_24/ReadVariableOp_1+lstm_24/while/lstm_cell_24/ReadVariableOp_12Z
+lstm_24/while/lstm_cell_24/ReadVariableOp_2+lstm_24/while/lstm_cell_24/ReadVariableOp_22Z
+lstm_24/while/lstm_cell_24/ReadVariableOp_3+lstm_24/while/lstm_cell_24/ReadVariableOp_32b
/lstm_24/while/lstm_cell_24/split/ReadVariableOp/lstm_24/while/lstm_cell_24/split/ReadVariableOp2f
1lstm_24/while/lstm_cell_24/split_1/ReadVariableOp1lstm_24/while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
Â

)__inference_dense_73_layer_call_fn_484691

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_481990o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü	
Ú
.__inference_sequential_23_layer_call_fn_482633

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity¢StatefulPartitionedCall¹
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_482013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
£
I__inference_sequential_23_layer_call_and_return_conditional_losses_482604
dense_72_input!
dense_72_482581:
dense_72_482583: 
lstm_24_482586:
lstm_24_482588: 
lstm_24_482590:!
dense_73_482593:
dense_73_482595:!
dense_74_482598:
dense_74_482600:
identity¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCallü
 dense_72/StatefulPartitionedCallStatefulPartitionedCalldense_72_inputdense_72_482581dense_72_482583*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_481724¡
lstm_24/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0lstm_24_482586lstm_24_482588lstm_24_482590*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_482438
 dense_73/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0dense_73_482593dense_73_482595*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_481990
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_482598dense_74_482600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_482006x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_72_input
v
	
while_body_484177
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_24_split_readvariableop_resource_0:B
4while_lstm_cell_24_split_1_readvariableop_resource_0:>
,while_lstm_cell_24_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_24_split_readvariableop_resource:@
2while_lstm_cell_24_split_1_readvariableop_resource:<
*while_lstm_cell_24_readvariableop_resource:¢!while/lstm_cell_24/ReadVariableOp¢#while/lstm_cell_24/ReadVariableOp_1¢#while/lstm_cell_24/ReadVariableOp_2¢#while/lstm_cell_24/ReadVariableOp_3¢'while/lstm_cell_24/split/ReadVariableOp¢)while/lstm_cell_24/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
"while/lstm_cell_24/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
while/lstm_cell_24/ones_likeFill+while/lstm_cell_24/ones_like/Shape:output:0+while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$while/lstm_cell_24/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
while/lstm_cell_24/ones_like_1Fill-while/lstm_cell_24/ones_like_1/Shape:output:0-while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
while/lstm_cell_24/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_24/split/ReadVariableOpReadVariableOp2while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0×
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
while/lstm_cell_24/MatMulMatMulwhile/lstm_cell_24/mul:z:0!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_1MatMulwhile/lstm_cell_24/mul_1:z:0!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_2MatMulwhile/lstm_cell_24/mul_2:z:0!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_3MatMulwhile/lstm_cell_24/mul_3:z:0!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0Í
while/lstm_cell_24/split_1Split-while/lstm_cell_24/split_1/split_dim:output:01while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¡
while/lstm_cell_24/BiasAddBiasAdd#while/lstm_cell_24/MatMul:product:0#while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_1BiasAdd%while/lstm_cell_24/MatMul_1:product:0#while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_2BiasAdd%while/lstm_cell_24/MatMul_2:product:0#while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_3BiasAdd%while/lstm_cell_24/MatMul_3:product:0#while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_4Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_5Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_6Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_7Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/ReadVariableOpReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 while/lstm_cell_24/strided_sliceStridedSlice)while/lstm_cell_24/ReadVariableOp:value:0/while/lstm_cell_24/strided_slice/stack:output:01while/lstm_cell_24/strided_slice/stack_1:output:01while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask 
while/lstm_cell_24/MatMul_4MatMulwhile/lstm_cell_24/mul_4:z:0)while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/addAddV2#while/lstm_cell_24/BiasAdd:output:0%while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_24/SigmoidSigmoidwhile/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_1ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_1StridedSlice+while/lstm_cell_24/ReadVariableOp_1:value:01while/lstm_cell_24/strided_slice_1/stack:output:03while/lstm_cell_24/strided_slice_1/stack_1:output:03while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_5MatMulwhile/lstm_cell_24/mul_5:z:0+while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_1AddV2%while/lstm_cell_24/BiasAdd_1:output:0%while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_1Sigmoidwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_8Mul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_2ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_2StridedSlice+while/lstm_cell_24/ReadVariableOp_2:value:01while/lstm_cell_24/strided_slice_2/stack:output:03while/lstm_cell_24/strided_slice_2/stack_1:output:03while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_6MatMulwhile/lstm_cell_24/mul_6:z:0+while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_2AddV2%while/lstm_cell_24/BiasAdd_2:output:0%while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_2Sigmoidwhile/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_9Mulwhile/lstm_cell_24/Sigmoid:y:0 while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/add_3AddV2while/lstm_cell_24/mul_8:z:0while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_3ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_3StridedSlice+while/lstm_cell_24/ReadVariableOp_3:value:01while/lstm_cell_24/strided_slice_3/stack:output:03while/lstm_cell_24/strided_slice_3/stack_1:output:03while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_7MatMulwhile/lstm_cell_24/mul_7:z:0+while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_4AddV2%while/lstm_cell_24/BiasAdd_3:output:0%while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_3Sigmoidwhile/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_4Sigmoidwhile/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_10Mul while/lstm_cell_24/Sigmoid_3:y:0 while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_24/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_24/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸

while/NoOpNoOp"^while/lstm_cell_24/ReadVariableOp$^while/lstm_cell_24/ReadVariableOp_1$^while/lstm_cell_24/ReadVariableOp_2$^while/lstm_cell_24/ReadVariableOp_3(^while/lstm_cell_24/split/ReadVariableOp*^while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_24_readvariableop_resource,while_lstm_cell_24_readvariableop_resource_0"j
2while_lstm_cell_24_split_1_readvariableop_resource4while_lstm_cell_24_split_1_readvariableop_resource_0"f
0while_lstm_cell_24_split_readvariableop_resource2while_lstm_cell_24_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2F
!while/lstm_cell_24/ReadVariableOp!while/lstm_cell_24/ReadVariableOp2J
#while/lstm_cell_24/ReadVariableOp_1#while/lstm_cell_24/ReadVariableOp_12J
#while/lstm_cell_24/ReadVariableOp_2#while/lstm_cell_24/ReadVariableOp_22J
#while/lstm_cell_24/ReadVariableOp_3#while/lstm_cell_24/ReadVariableOp_32R
'while/lstm_cell_24/split/ReadVariableOp'while/lstm_cell_24/split/ReadVariableOp2V
)while/lstm_cell_24/split_1/ReadVariableOp)while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
¤

I__inference_sequential_23_layer_call_and_return_conditional_losses_482013

inputs!
dense_72_481725:
dense_72_481727: 
lstm_24_481973:
lstm_24_481975: 
lstm_24_481977:!
dense_73_481991:
dense_73_481993:!
dense_74_482007:
dense_74_482009:
identity¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCallô
 dense_72/StatefulPartitionedCallStatefulPartitionedCallinputsdense_72_481725dense_72_481727*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_481724¡
lstm_24/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0lstm_24_481973lstm_24_481975lstm_24_481977*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_481972
 dense_73/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0dense_73_481991dense_73_481993*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_481990
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_482007dense_74_482009*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_482006x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó"
Ý
while_body_481609
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_24_481633_0:)
while_lstm_cell_24_481635_0:-
while_lstm_cell_24_481637_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_24_481633:'
while_lstm_cell_24_481635:+
while_lstm_cell_24_481637:¢*while/lstm_cell_24/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_24_481633_0while_lstm_cell_24_481635_0while_lstm_cell_24_481637_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481550Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_24/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_24/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_24_481633while_lstm_cell_24_481633_0"8
while_lstm_cell_24_481635while_lstm_cell_24_481635_0"8
while_lstm_cell_24_481637while_lstm_cell_24_481637_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_24/StatefulPartitionedCall*while/lstm_cell_24/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
ÿÉ
ä
C__inference_lstm_24_layer_call_and_return_conditional_losses_484682

inputs<
*lstm_cell_24_split_readvariableop_resource::
,lstm_cell_24_split_1_readvariableop_resource:6
$lstm_cell_24_readvariableop_resource:
identity¢lstm_cell_24/ReadVariableOp¢lstm_cell_24/ReadVariableOp_1¢lstm_cell_24/ReadVariableOp_2¢lstm_cell_24/ReadVariableOp_3¢!lstm_cell_24/split/ReadVariableOp¢#lstm_cell_24/split_1/ReadVariableOp¢while;
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskd
lstm_cell_24/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_24/ones_likeFill%lstm_cell_24/ones_like/Shape:output:0%lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout/MulMullstm_cell_24/ones_like:output:0#lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_24/dropout/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¦
1lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0h
#lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Í
!lstm_cell_24/dropout/GreaterEqualGreaterEqual:lstm_cell_24/dropout/random_uniform/RandomUniform:output:0,lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout/CastCast%lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout/Mul_1Mullstm_cell_24/dropout/Mul:z:0lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_1/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_1/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_1/GreaterEqualGreaterEqual<lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_1/CastCast'lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_1/Mul_1Mullstm_cell_24/dropout_1/Mul:z:0lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_2/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_2/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_2/GreaterEqualGreaterEqual<lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_2/CastCast'lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_2/Mul_1Mullstm_cell_24/dropout_2/Mul:z:0lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_3/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_3/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_3/GreaterEqualGreaterEqual<lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_3/CastCast'lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_3/Mul_1Mullstm_cell_24/dropout_3/Mul:z:0lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
lstm_cell_24/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm_cell_24/ones_like_1Fill'lstm_cell_24/ones_like_1/Shape:output:0'lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_4/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_4/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_4/GreaterEqualGreaterEqual<lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_4/CastCast'lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_4/Mul_1Mullstm_cell_24/dropout_4/Mul:z:0lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_5/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_5/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_5/GreaterEqualGreaterEqual<lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_5/CastCast'lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_5/Mul_1Mullstm_cell_24/dropout_5/Mul:z:0lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_6/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_6/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_6/GreaterEqualGreaterEqual<lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_6/CastCast'lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_6/Mul_1Mullstm_cell_24/dropout_6/Mul:z:0lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_7/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_7/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_7/GreaterEqualGreaterEqual<lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_7/CastCast'lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_7/Mul_1Mullstm_cell_24/dropout_7/Mul:z:0lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mulMulstrided_slice_2:output:0lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_1Mulstrided_slice_2:output:0 lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_2Mulstrided_slice_2:output:0 lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_3Mulstrided_slice_2:output:0 lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_24/split/ReadVariableOpReadVariableOp*lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Å
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0)lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_cell_24/MatMulMatMullstm_cell_24/mul:z:0lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_1MatMullstm_cell_24/mul_1:z:0lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_2MatMullstm_cell_24/mul_2:z:0lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_3MatMullstm_cell_24/mul_3:z:0lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_24/split_1/ReadVariableOpReadVariableOp,lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0»
lstm_cell_24/split_1Split'lstm_cell_24/split_1/split_dim:output:0+lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
lstm_cell_24/BiasAddBiasAddlstm_cell_24/MatMul:product:0lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_1BiasAddlstm_cell_24/MatMul_1:product:0lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_2BiasAddlstm_cell_24/MatMul_2:product:0lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_3BiasAddlstm_cell_24/MatMul_3:product:0lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_4Mulzeros:output:0 lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_5Mulzeros:output:0 lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_6Mulzeros:output:0 lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_7Mulzeros:output:0 lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOpReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¬
lstm_cell_24/strided_sliceStridedSlice#lstm_cell_24/ReadVariableOp:value:0)lstm_cell_24/strided_slice/stack:output:0+lstm_cell_24/strided_slice/stack_1:output:0+lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_4MatMullstm_cell_24/mul_4:z:0#lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/addAddV2lstm_cell_24/BiasAdd:output:0lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_24/SigmoidSigmoidlstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_1ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_1StridedSlice%lstm_cell_24/ReadVariableOp_1:value:0+lstm_cell_24/strided_slice_1/stack:output:0-lstm_cell_24/strided_slice_1/stack_1:output:0-lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_5MatMullstm_cell_24/mul_5:z:0%lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_1AddV2lstm_cell_24/BiasAdd_1:output:0lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_24/mul_8Mullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_2ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_2StridedSlice%lstm_cell_24/ReadVariableOp_2:value:0+lstm_cell_24/strided_slice_2/stack:output:0-lstm_cell_24/strided_slice_2/stack_1:output:0-lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_6MatMullstm_cell_24/mul_6:z:0%lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_2AddV2lstm_cell_24/BiasAdd_2:output:0lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_9Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/add_3AddV2lstm_cell_24/mul_8:z:0lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_3ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_3StridedSlice%lstm_cell_24/ReadVariableOp_3:value:0+lstm_cell_24/strided_slice_3/stack:output:0-lstm_cell_24/strided_slice_3/stack_1:output:0-lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_7MatMullstm_cell_24/mul_7:z:0%lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_4AddV2lstm_cell_24/BiasAdd_3:output:0lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_3Sigmoidlstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_4Sigmoidlstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_10Mullstm_cell_24/Sigmoid_3:y:0lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_24_split_readvariableop_resource,lstm_cell_24_split_1_readvariableop_resource$lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_484484*
condR
while_cond_484483*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell_24/ReadVariableOp^lstm_cell_24/ReadVariableOp_1^lstm_cell_24/ReadVariableOp_2^lstm_cell_24/ReadVariableOp_3"^lstm_cell_24/split/ReadVariableOp$^lstm_cell_24/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2:
lstm_cell_24/ReadVariableOplstm_cell_24/ReadVariableOp2>
lstm_cell_24/ReadVariableOp_1lstm_cell_24/ReadVariableOp_12>
lstm_cell_24/ReadVariableOp_2lstm_cell_24/ReadVariableOp_22>
lstm_cell_24/ReadVariableOp_3lstm_cell_24/ReadVariableOp_32F
!lstm_cell_24/split/ReadVariableOp!lstm_cell_24/split/ReadVariableOp2J
#lstm_cell_24/split_1/ReadVariableOp#lstm_cell_24/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¤

I__inference_sequential_23_layer_call_and_return_conditional_losses_482508

inputs!
dense_72_482485:
dense_72_482487: 
lstm_24_482490:
lstm_24_482492: 
lstm_24_482494:!
dense_73_482497:
dense_73_482499:!
dense_74_482502:
dense_74_482504:
identity¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCallô
 dense_72/StatefulPartitionedCallStatefulPartitionedCallinputsdense_72_482485dense_72_482487*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_481724¡
lstm_24/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0lstm_24_482490lstm_24_482492lstm_24_482494*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_482438
 dense_73/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0dense_73_482497dense_73_482499*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_481990
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_482502dense_74_482504*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_482006x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
v
	
while_body_481838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0D
2while_lstm_cell_24_split_readvariableop_resource_0:B
4while_lstm_cell_24_split_1_readvariableop_resource_0:>
,while_lstm_cell_24_readvariableop_resource_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorB
0while_lstm_cell_24_split_readvariableop_resource:@
2while_lstm_cell_24_split_1_readvariableop_resource:<
*while_lstm_cell_24_readvariableop_resource:¢!while/lstm_cell_24/ReadVariableOp¢#while/lstm_cell_24/ReadVariableOp_1¢#while/lstm_cell_24/ReadVariableOp_2¢#while/lstm_cell_24/ReadVariableOp_3¢'while/lstm_cell_24/split/ReadVariableOp¢)while/lstm_cell_24/split_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
"while/lstm_cell_24/ones_like/ShapeShape0while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:g
"while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?°
while/lstm_cell_24/ones_likeFill+while/lstm_cell_24/ones_like/Shape:output:0+while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
$while/lstm_cell_24/ones_like_1/ShapeShapewhile_placeholder_2*
T0*
_output_shapes
:i
$while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¶
while/lstm_cell_24/ones_like_1Fill-while/lstm_cell_24/ones_like_1/Shape:output:0-while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
while/lstm_cell_24/mulMul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_1Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_2Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿª
while/lstm_cell_24/mul_3Mul0while/TensorArrayV2Read/TensorListGetItem:item:0%while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
'while/lstm_cell_24/split/ReadVariableOpReadVariableOp2while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0×
while/lstm_cell_24/splitSplit+while/lstm_cell_24/split/split_dim:output:0/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
while/lstm_cell_24/MatMulMatMulwhile/lstm_cell_24/mul:z:0!while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_1MatMulwhile/lstm_cell_24/mul_1:z:0!while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_2MatMulwhile/lstm_cell_24/mul_2:z:0!while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/MatMul_3MatMulwhile/lstm_cell_24/mul_3:z:0!while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
$while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
)while/lstm_cell_24/split_1/ReadVariableOpReadVariableOp4while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0Í
while/lstm_cell_24/split_1Split-while/lstm_cell_24/split_1/split_dim:output:01while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split¡
while/lstm_cell_24/BiasAddBiasAdd#while/lstm_cell_24/MatMul:product:0#while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_1BiasAdd%while/lstm_cell_24/MatMul_1:product:0#while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_2BiasAdd%while/lstm_cell_24/MatMul_2:product:0#while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¥
while/lstm_cell_24/BiasAdd_3BiasAdd%while/lstm_cell_24/MatMul_3:product:0#while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_4Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_5Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_6Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_7Mulwhile_placeholder_2'while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!while/lstm_cell_24/ReadVariableOpReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0w
&while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        y
(while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       y
(while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ê
 while/lstm_cell_24/strided_sliceStridedSlice)while/lstm_cell_24/ReadVariableOp:value:0/while/lstm_cell_24/strided_slice/stack:output:01while/lstm_cell_24/strided_slice/stack_1:output:01while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask 
while/lstm_cell_24/MatMul_4MatMulwhile/lstm_cell_24/mul_4:z:0)while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/addAddV2#while/lstm_cell_24/BiasAdd:output:0%while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
while/lstm_cell_24/SigmoidSigmoidwhile/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_1ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_1StridedSlice+while/lstm_cell_24/ReadVariableOp_1:value:01while/lstm_cell_24/strided_slice_1/stack:output:03while/lstm_cell_24/strided_slice_1/stack_1:output:03while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_5MatMulwhile/lstm_cell_24/mul_5:z:0+while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_1AddV2%while/lstm_cell_24/BiasAdd_1:output:0%while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_1Sigmoidwhile/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_8Mul while/lstm_cell_24/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_2ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_2StridedSlice+while/lstm_cell_24/ReadVariableOp_2:value:01while/lstm_cell_24/strided_slice_2/stack:output:03while/lstm_cell_24/strided_slice_2/stack_1:output:03while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_6MatMulwhile/lstm_cell_24/mul_6:z:0+while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_2AddV2%while/lstm_cell_24/BiasAdd_2:output:0%while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_2Sigmoidwhile/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_9Mulwhile/lstm_cell_24/Sigmoid:y:0 while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/add_3AddV2while/lstm_cell_24/mul_8:z:0while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#while/lstm_cell_24/ReadVariableOp_3ReadVariableOp,while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0y
(while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       {
*while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        {
*while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Ô
"while/lstm_cell_24/strided_slice_3StridedSlice+while/lstm_cell_24/ReadVariableOp_3:value:01while/lstm_cell_24/strided_slice_3/stack:output:03while/lstm_cell_24/strided_slice_3/stack_1:output:03while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask¢
while/lstm_cell_24/MatMul_7MatMulwhile/lstm_cell_24/mul_7:z:0+while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
while/lstm_cell_24/add_4AddV2%while/lstm_cell_24/BiasAdd_3:output:0%while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_3Sigmoidwhile/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
while/lstm_cell_24/Sigmoid_4Sigmoidwhile/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_24/mul_10Mul while/lstm_cell_24/Sigmoid_3:y:0 while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÆ
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholderwhile/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒz
while/Identity_4Identitywhile/lstm_cell_24/mul_10:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
while/Identity_5Identitywhile/lstm_cell_24/add_3:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¸

while/NoOpNoOp"^while/lstm_cell_24/ReadVariableOp$^while/lstm_cell_24/ReadVariableOp_1$^while/lstm_cell_24/ReadVariableOp_2$^while/lstm_cell_24/ReadVariableOp_3(^while/lstm_cell_24/split/ReadVariableOp*^while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"Z
*while_lstm_cell_24_readvariableop_resource,while_lstm_cell_24_readvariableop_resource_0"j
2while_lstm_cell_24_split_1_readvariableop_resource4while_lstm_cell_24_split_1_readvariableop_resource_0"f
0while_lstm_cell_24_split_readvariableop_resource2while_lstm_cell_24_split_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2F
!while/lstm_cell_24/ReadVariableOp!while/lstm_cell_24/ReadVariableOp2J
#while/lstm_cell_24/ReadVariableOp_1#while/lstm_cell_24/ReadVariableOp_12J
#while/lstm_cell_24/ReadVariableOp_2#while/lstm_cell_24/ReadVariableOp_22J
#while/lstm_cell_24/ReadVariableOp_3#while/lstm_cell_24/ReadVariableOp_32R
'while/lstm_cell_24/split/ReadVariableOp'while/lstm_cell_24/split/ReadVariableOp2V
)while/lstm_cell_24/split_1/ReadVariableOp)while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_481303
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_481303___redundant_placeholder04
0while_while_cond_481303___redundant_placeholder14
0while_while_cond_481303___redundant_placeholder24
0while_while_cond_481303___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
ó"
Ý
while_body_481304
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_24_481328_0:)
while_lstm_cell_24_481330_0:-
while_lstm_cell_24_481332_0:
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_24_481328:'
while_lstm_cell_24_481330:+
while_lstm_cell_24_481332:¢*while/lstm_cell_24/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0³
*while/lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_24_481328_0while_lstm_cell_24_481330_0while_lstm_cell_24_481332_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481290Ü
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_1while_placeholder3while/lstm_cell_24/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
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
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: :éèÒ
while/Identity_4Identity3while/lstm_cell_24/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/Identity_5Identity3while/lstm_cell_24/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy

while/NoOpNoOp+^while/lstm_cell_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"8
while_lstm_cell_24_481328while_lstm_cell_24_481328_0"8
while_lstm_cell_24_481330while_lstm_cell_24_481330_0"8
while_lstm_cell_24_481332while_lstm_cell_24_481332_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2X
*while/lstm_cell_24/StatefulPartitionedCall*while/lstm_cell_24/StatefulPartitionedCall: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
8

C__inference_lstm_24_layer_call_and_return_conditional_losses_481678

inputs%
lstm_cell_24_481596:!
lstm_cell_24_481598:%
lstm_cell_24_481600:
identity¢$lstm_cell_24/StatefulPartitionedCall¢while;
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_24_481596lstm_cell_24_481598lstm_cell_24_481600*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481550n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_24_481596lstm_cell_24_481598lstm_cell_24_481600*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_481609*
condR
while_cond_481608*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_24/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_24/StatefulPartitionedCall$lstm_cell_24/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


â
.__inference_sequential_23_layer_call_fn_482552
dense_72_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCalldense_72_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_482508o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_72_input
ô
²
(__inference_lstm_24_layer_call_fn_483454

inputs
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallå
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_482438o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_483562
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_483562___redundant_placeholder04
0while_while_cond_483562___redundant_placeholder14
0while_while_cond_483562___redundant_placeholder24
0while_while_cond_483562___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
è
ó
-__inference_lstm_cell_24_layer_call_fn_484737

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481290o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Â

)__inference_dense_74_layer_call_fn_484710

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÙ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_482006o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

û
'sequential_23_lstm_24_while_cond_481026H
Dsequential_23_lstm_24_while_sequential_23_lstm_24_while_loop_counterN
Jsequential_23_lstm_24_while_sequential_23_lstm_24_while_maximum_iterations+
'sequential_23_lstm_24_while_placeholder-
)sequential_23_lstm_24_while_placeholder_1-
)sequential_23_lstm_24_while_placeholder_2-
)sequential_23_lstm_24_while_placeholder_3J
Fsequential_23_lstm_24_while_less_sequential_23_lstm_24_strided_slice_1`
\sequential_23_lstm_24_while_sequential_23_lstm_24_while_cond_481026___redundant_placeholder0`
\sequential_23_lstm_24_while_sequential_23_lstm_24_while_cond_481026___redundant_placeholder1`
\sequential_23_lstm_24_while_sequential_23_lstm_24_while_cond_481026___redundant_placeholder2`
\sequential_23_lstm_24_while_sequential_23_lstm_24_while_cond_481026___redundant_placeholder3(
$sequential_23_lstm_24_while_identity
º
 sequential_23/lstm_24/while/LessLess'sequential_23_lstm_24_while_placeholderFsequential_23_lstm_24_while_less_sequential_23_lstm_24_strided_slice_1*
T0*
_output_shapes
: w
$sequential_23/lstm_24/while/IdentityIdentity$sequential_23/lstm_24/while/Less:z:0*
T0
*
_output_shapes
: "U
$sequential_23_lstm_24_while_identity-sequential_23/lstm_24/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


ã
lstm_24_while_cond_482790,
(lstm_24_while_lstm_24_while_loop_counter2
.lstm_24_while_lstm_24_while_maximum_iterations
lstm_24_while_placeholder
lstm_24_while_placeholder_1
lstm_24_while_placeholder_2
lstm_24_while_placeholder_3.
*lstm_24_while_less_lstm_24_strided_slice_1D
@lstm_24_while_lstm_24_while_cond_482790___redundant_placeholder0D
@lstm_24_while_lstm_24_while_cond_482790___redundant_placeholder1D
@lstm_24_while_lstm_24_while_cond_482790___redundant_placeholder2D
@lstm_24_while_lstm_24_while_cond_482790___redundant_placeholder3
lstm_24_while_identity

lstm_24/while/LessLesslstm_24_while_placeholder*lstm_24_while_less_lstm_24_strided_slice_1*
T0*
_output_shapes
: [
lstm_24/while/IdentityIdentitylstm_24/while/Less:z:0*
T0
*
_output_shapes
: "9
lstm_24_while_identitylstm_24/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:
½~
¦
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481550

inputs

states
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿG
ones_like_1/ShapeShapestates*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_4Mulstatesdropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_5Mulstatesdropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_6Mulstatesdropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_7Mulstatesdropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
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
valueB"      ë
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
:ÿÿÿÿÿÿÿÿÿd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_namestates
Ë
û
D__inference_dense_72_layer_call_and_return_conditional_losses_483410

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
value	B : »
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
value	B : ¿
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
value	B : 
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
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

´
(__inference_lstm_24_layer_call_fn_483421
inputs_0
unknown:
	unknown_0:
	unknown_1:
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_481373o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
Ë
û
D__inference_dense_72_layer_call_and_return_conditional_losses_481724

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
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
value	B : »
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
value	B : ¿
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
value	B : 
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
:ÿÿÿÿÿÿÿÿÿ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : §
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ÿÉ
ä
C__inference_lstm_24_layer_call_and_return_conditional_losses_482438

inputs<
*lstm_cell_24_split_readvariableop_resource::
,lstm_cell_24_split_1_readvariableop_resource:6
$lstm_cell_24_readvariableop_resource:
identity¢lstm_cell_24/ReadVariableOp¢lstm_cell_24/ReadVariableOp_1¢lstm_cell_24/ReadVariableOp_2¢lstm_cell_24/ReadVariableOp_3¢!lstm_cell_24/split/ReadVariableOp¢#lstm_cell_24/split_1/ReadVariableOp¢while;
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskd
lstm_cell_24/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_24/ones_likeFill%lstm_cell_24/ones_like/Shape:output:0%lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout/MulMullstm_cell_24/ones_like:output:0#lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_24/dropout/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¦
1lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0h
#lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Í
!lstm_cell_24/dropout/GreaterEqualGreaterEqual:lstm_cell_24/dropout/random_uniform/RandomUniform:output:0,lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout/CastCast%lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout/Mul_1Mullstm_cell_24/dropout/Mul:z:0lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_1/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_1/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_1/GreaterEqualGreaterEqual<lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_1/CastCast'lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_1/Mul_1Mullstm_cell_24/dropout_1/Mul:z:0lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_2/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_2/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_2/GreaterEqualGreaterEqual<lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_2/CastCast'lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_2/Mul_1Mullstm_cell_24/dropout_2/Mul:z:0lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_3/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_3/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_3/GreaterEqualGreaterEqual<lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_3/CastCast'lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_3/Mul_1Mullstm_cell_24/dropout_3/Mul:z:0lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
lstm_cell_24/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm_cell_24/ones_like_1Fill'lstm_cell_24/ones_like_1/Shape:output:0'lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_4/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_4/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_4/GreaterEqualGreaterEqual<lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_4/CastCast'lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_4/Mul_1Mullstm_cell_24/dropout_4/Mul:z:0lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_5/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_5/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_5/GreaterEqualGreaterEqual<lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_5/CastCast'lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_5/Mul_1Mullstm_cell_24/dropout_5/Mul:z:0lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_6/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_6/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_6/GreaterEqualGreaterEqual<lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_6/CastCast'lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_6/Mul_1Mullstm_cell_24/dropout_6/Mul:z:0lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_7/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_7/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_7/GreaterEqualGreaterEqual<lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_7/CastCast'lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_7/Mul_1Mullstm_cell_24/dropout_7/Mul:z:0lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mulMulstrided_slice_2:output:0lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_1Mulstrided_slice_2:output:0 lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_2Mulstrided_slice_2:output:0 lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_3Mulstrided_slice_2:output:0 lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_24/split/ReadVariableOpReadVariableOp*lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Å
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0)lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_cell_24/MatMulMatMullstm_cell_24/mul:z:0lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_1MatMullstm_cell_24/mul_1:z:0lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_2MatMullstm_cell_24/mul_2:z:0lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_3MatMullstm_cell_24/mul_3:z:0lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_24/split_1/ReadVariableOpReadVariableOp,lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0»
lstm_cell_24/split_1Split'lstm_cell_24/split_1/split_dim:output:0+lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
lstm_cell_24/BiasAddBiasAddlstm_cell_24/MatMul:product:0lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_1BiasAddlstm_cell_24/MatMul_1:product:0lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_2BiasAddlstm_cell_24/MatMul_2:product:0lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_3BiasAddlstm_cell_24/MatMul_3:product:0lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_4Mulzeros:output:0 lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_5Mulzeros:output:0 lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_6Mulzeros:output:0 lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_7Mulzeros:output:0 lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOpReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¬
lstm_cell_24/strided_sliceStridedSlice#lstm_cell_24/ReadVariableOp:value:0)lstm_cell_24/strided_slice/stack:output:0+lstm_cell_24/strided_slice/stack_1:output:0+lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_4MatMullstm_cell_24/mul_4:z:0#lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/addAddV2lstm_cell_24/BiasAdd:output:0lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_24/SigmoidSigmoidlstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_1ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_1StridedSlice%lstm_cell_24/ReadVariableOp_1:value:0+lstm_cell_24/strided_slice_1/stack:output:0-lstm_cell_24/strided_slice_1/stack_1:output:0-lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_5MatMullstm_cell_24/mul_5:z:0%lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_1AddV2lstm_cell_24/BiasAdd_1:output:0lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_24/mul_8Mullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_2ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_2StridedSlice%lstm_cell_24/ReadVariableOp_2:value:0+lstm_cell_24/strided_slice_2/stack:output:0-lstm_cell_24/strided_slice_2/stack_1:output:0-lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_6MatMullstm_cell_24/mul_6:z:0%lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_2AddV2lstm_cell_24/BiasAdd_2:output:0lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_9Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/add_3AddV2lstm_cell_24/mul_8:z:0lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_3ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_3StridedSlice%lstm_cell_24/ReadVariableOp_3:value:0+lstm_cell_24/strided_slice_3/stack:output:0-lstm_cell_24/strided_slice_3/stack_1:output:0-lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_7MatMullstm_cell_24/mul_7:z:0%lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_4AddV2lstm_cell_24/BiasAdd_3:output:0lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_3Sigmoidlstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_4Sigmoidlstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_10Mullstm_cell_24/Sigmoid_3:y:0lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_24_split_readvariableop_resource,lstm_cell_24_split_1_readvariableop_resource$lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_482240*
condR
while_cond_482239*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Â
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell_24/ReadVariableOp^lstm_cell_24/ReadVariableOp_1^lstm_cell_24/ReadVariableOp_2^lstm_cell_24/ReadVariableOp_3"^lstm_cell_24/split/ReadVariableOp$^lstm_cell_24/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2:
lstm_cell_24/ReadVariableOplstm_cell_24/ReadVariableOp2>
lstm_cell_24/ReadVariableOp_1lstm_cell_24/ReadVariableOp_12>
lstm_cell_24/ReadVariableOp_2lstm_cell_24/ReadVariableOp_22>
lstm_cell_24/ReadVariableOp_3lstm_cell_24/ReadVariableOp_32F
!lstm_cell_24/split/ReadVariableOp!lstm_cell_24/split/ReadVariableOp2J
#lstm_cell_24/split_1/ReadVariableOp#lstm_cell_24/split_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
´Ê
æ
C__inference_lstm_24_layer_call_and_return_conditional_losses_484068
inputs_0<
*lstm_cell_24_split_readvariableop_resource::
,lstm_cell_24_split_1_readvariableop_resource:6
$lstm_cell_24_readvariableop_resource:
identity¢lstm_cell_24/ReadVariableOp¢lstm_cell_24/ReadVariableOp_1¢lstm_cell_24/ReadVariableOp_2¢lstm_cell_24/ReadVariableOp_3¢!lstm_cell_24/split/ReadVariableOp¢#lstm_cell_24/split_1/ReadVariableOp¢while=
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskd
lstm_cell_24/ones_like/ShapeShapestrided_slice_2:output:0*
T0*
_output_shapes
:a
lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
lstm_cell_24/ones_likeFill%lstm_cell_24/ones_like/Shape:output:0%lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
lstm_cell_24/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout/MulMullstm_cell_24/ones_like:output:0#lstm_cell_24/dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
lstm_cell_24/dropout/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:¦
1lstm_cell_24/dropout/random_uniform/RandomUniformRandomUniform#lstm_cell_24/dropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0h
#lstm_cell_24/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Í
!lstm_cell_24/dropout/GreaterEqualGreaterEqual:lstm_cell_24/dropout/random_uniform/RandomUniform:output:0,lstm_cell_24/dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout/CastCast%lstm_cell_24/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout/Mul_1Mullstm_cell_24/dropout/Mul:z:0lstm_cell_24/dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_1/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_1/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_1/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_1/GreaterEqualGreaterEqual<lstm_cell_24/dropout_1/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_1/CastCast'lstm_cell_24/dropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_1/Mul_1Mullstm_cell_24/dropout_1/Mul:z:0lstm_cell_24/dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_2/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_2/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_2/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_2/GreaterEqualGreaterEqual<lstm_cell_24/dropout_2/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_2/CastCast'lstm_cell_24/dropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_2/Mul_1Mullstm_cell_24/dropout_2/Mul:z:0lstm_cell_24/dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_3/MulMullstm_cell_24/ones_like:output:0%lstm_cell_24/dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/dropout_3/ShapeShapelstm_cell_24/ones_like:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_3/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_3/GreaterEqualGreaterEqual<lstm_cell_24/dropout_3/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_3/CastCast'lstm_cell_24/dropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_3/Mul_1Mullstm_cell_24/dropout_3/Mul:z:0lstm_cell_24/dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ\
lstm_cell_24/ones_like_1/ShapeShapezeros:output:0*
T0*
_output_shapes
:c
lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?¤
lstm_cell_24/ones_like_1Fill'lstm_cell_24/ones_like_1/Shape:output:0'lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_4/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_4/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_4/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_4/GreaterEqualGreaterEqual<lstm_cell_24/dropout_4/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_4/CastCast'lstm_cell_24/dropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_4/Mul_1Mullstm_cell_24/dropout_4/Mul:z:0lstm_cell_24/dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_5/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_5/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_5/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_5/GreaterEqualGreaterEqual<lstm_cell_24/dropout_5/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_5/CastCast'lstm_cell_24/dropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_5/Mul_1Mullstm_cell_24/dropout_5/Mul:z:0lstm_cell_24/dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_6/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_6/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_6/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_6/GreaterEqualGreaterEqual<lstm_cell_24/dropout_6/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_6/CastCast'lstm_cell_24/dropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_6/Mul_1Mullstm_cell_24/dropout_6/Mul:z:0lstm_cell_24/dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
lstm_cell_24/dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?
lstm_cell_24/dropout_7/MulMul!lstm_cell_24/ones_like_1:output:0%lstm_cell_24/dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
lstm_cell_24/dropout_7/ShapeShape!lstm_cell_24/ones_like_1:output:0*
T0*
_output_shapes
:ª
3lstm_cell_24/dropout_7/random_uniform/RandomUniformRandomUniform%lstm_cell_24/dropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0j
%lstm_cell_24/dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=Ó
#lstm_cell_24/dropout_7/GreaterEqualGreaterEqual<lstm_cell_24/dropout_7/random_uniform/RandomUniform:output:0.lstm_cell_24/dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_7/CastCast'lstm_cell_24/dropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/dropout_7/Mul_1Mullstm_cell_24/dropout_7/Mul:z:0lstm_cell_24/dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mulMulstrided_slice_2:output:0lstm_cell_24/dropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_1Mulstrided_slice_2:output:0 lstm_cell_24/dropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_2Mulstrided_slice_2:output:0 lstm_cell_24/dropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_3Mulstrided_slice_2:output:0 lstm_cell_24/dropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
!lstm_cell_24/split/ReadVariableOpReadVariableOp*lstm_cell_24_split_readvariableop_resource*
_output_shapes

:*
dtype0Å
lstm_cell_24/splitSplit%lstm_cell_24/split/split_dim:output:0)lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split
lstm_cell_24/MatMulMatMullstm_cell_24/mul:z:0lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_1MatMullstm_cell_24/mul_1:z:0lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_2MatMullstm_cell_24/mul_2:z:0lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/MatMul_3MatMullstm_cell_24/mul_3:z:0lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
#lstm_cell_24/split_1/ReadVariableOpReadVariableOp,lstm_cell_24_split_1_readvariableop_resource*
_output_shapes
:*
dtype0»
lstm_cell_24/split_1Split'lstm_cell_24/split_1/split_dim:output:0+lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_split
lstm_cell_24/BiasAddBiasAddlstm_cell_24/MatMul:product:0lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_1BiasAddlstm_cell_24/MatMul_1:product:0lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_2BiasAddlstm_cell_24/MatMul_2:product:0lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/BiasAdd_3BiasAddlstm_cell_24/MatMul_3:product:0lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_4Mulzeros:output:0 lstm_cell_24/dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_5Mulzeros:output:0 lstm_cell_24/dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_6Mulzeros:output:0 lstm_cell_24/dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/mul_7Mulzeros:output:0 lstm_cell_24/dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOpReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0q
 lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        s
"lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       s
"lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¬
lstm_cell_24/strided_sliceStridedSlice#lstm_cell_24/ReadVariableOp:value:0)lstm_cell_24/strided_slice/stack:output:0+lstm_cell_24/strided_slice/stack_1:output:0+lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_4MatMullstm_cell_24/mul_4:z:0#lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/addAddV2lstm_cell_24/BiasAdd:output:0lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
lstm_cell_24/SigmoidSigmoidlstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_1ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_1StridedSlice%lstm_cell_24/ReadVariableOp_1:value:0+lstm_cell_24/strided_slice_1/stack:output:0-lstm_cell_24/strided_slice_1/stack_1:output:0-lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_5MatMullstm_cell_24/mul_5:z:0%lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_1AddV2lstm_cell_24/BiasAdd_1:output:0lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_1Sigmoidlstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
lstm_cell_24/mul_8Mullstm_cell_24/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_2ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_2StridedSlice%lstm_cell_24/ReadVariableOp_2:value:0+lstm_cell_24/strided_slice_2/stack:output:0-lstm_cell_24/strided_slice_2/stack_1:output:0-lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_6MatMullstm_cell_24/mul_6:z:0%lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_2AddV2lstm_cell_24/BiasAdd_2:output:0lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_2Sigmoidlstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_9Mullstm_cell_24/Sigmoid:y:0lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
lstm_cell_24/add_3AddV2lstm_cell_24/mul_8:z:0lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/ReadVariableOp_3ReadVariableOp$lstm_cell_24_readvariableop_resource*
_output_shapes

:*
dtype0s
"lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       u
$lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        u
$lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¶
lstm_cell_24/strided_slice_3StridedSlice%lstm_cell_24/ReadVariableOp_3:value:0+lstm_cell_24/strided_slice_3/stack:output:0-lstm_cell_24/strided_slice_3/stack_1:output:0-lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_mask
lstm_cell_24/MatMul_7MatMullstm_cell_24/mul_7:z:0%lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/add_4AddV2lstm_cell_24/BiasAdd_3:output:0lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_3Sigmoidlstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿk
lstm_cell_24/Sigmoid_4Sigmoidlstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_24/mul_10Mullstm_cell_24/Sigmoid_3:y:0lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ø
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_24_split_readvariableop_resource,lstm_cell_24_split_1_readvariableop_resource$lstm_cell_24_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_483870*
condR
while_cond_483869*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^lstm_cell_24/ReadVariableOp^lstm_cell_24/ReadVariableOp_1^lstm_cell_24/ReadVariableOp_2^lstm_cell_24/ReadVariableOp_3"^lstm_cell_24/split/ReadVariableOp$^lstm_cell_24/split_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2:
lstm_cell_24/ReadVariableOplstm_cell_24/ReadVariableOp2>
lstm_cell_24/ReadVariableOp_1lstm_cell_24/ReadVariableOp_12>
lstm_cell_24/ReadVariableOp_2lstm_cell_24/ReadVariableOp_22>
lstm_cell_24/ReadVariableOp_3lstm_cell_24/ReadVariableOp_32F
!lstm_cell_24/split/ReadVariableOp!lstm_cell_24/split/ReadVariableOp2J
#lstm_cell_24/split_1/ReadVariableOp#lstm_cell_24/split_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
¼
£
I__inference_sequential_23_layer_call_and_return_conditional_losses_482578
dense_72_input!
dense_72_482555:
dense_72_482557: 
lstm_24_482560:
lstm_24_482562: 
lstm_24_482564:!
dense_73_482567:
dense_73_482569:!
dense_74_482572:
dense_74_482574:
identity¢ dense_72/StatefulPartitionedCall¢ dense_73/StatefulPartitionedCall¢ dense_74/StatefulPartitionedCall¢lstm_24/StatefulPartitionedCallü
 dense_72/StatefulPartitionedCallStatefulPartitionedCalldense_72_inputdense_72_482555dense_72_482557*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_72_layer_call_and_return_conditional_losses_481724¡
lstm_24/StatefulPartitionedCallStatefulPartitionedCall)dense_72/StatefulPartitionedCall:output:0lstm_24_482560lstm_24_482562lstm_24_482564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_lstm_24_layer_call_and_return_conditional_losses_481972
 dense_73/StatefulPartitionedCallStatefulPartitionedCall(lstm_24/StatefulPartitionedCall:output:0dense_73_482567dense_73_482569*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_73_layer_call_and_return_conditional_losses_481990
 dense_74/StatefulPartitionedCallStatefulPartitionedCall)dense_73/StatefulPartitionedCall:output:0dense_74_482572dense_74_482574*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_dense_74_layer_call_and_return_conditional_losses_482006x
IdentityIdentity)dense_74/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
NoOpNoOp!^dense_72/StatefulPartitionedCall!^dense_73/StatefulPartitionedCall!^dense_74/StatefulPartitionedCall ^lstm_24/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 2D
 dense_72/StatefulPartitionedCall dense_72/StatefulPartitionedCall2D
 dense_73/StatefulPartitionedCall dense_73/StatefulPartitionedCall2D
 dense_74/StatefulPartitionedCall dense_74/StatefulPartitionedCall2B
lstm_24/StatefulPartitionedCalllstm_24/StatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_72_input
­D
¨
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_484836

inputs
states_0
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿX
mulMulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_1Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_2Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_3Mulinputsones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_4Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_5Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_6Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ^
mul_7Mulstates_0ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
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
valueB"      ë
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
:ÿÿÿÿÿÿÿÿÿd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Ü¥
¢
'sequential_23_lstm_24_while_body_481027H
Dsequential_23_lstm_24_while_sequential_23_lstm_24_while_loop_counterN
Jsequential_23_lstm_24_while_sequential_23_lstm_24_while_maximum_iterations+
'sequential_23_lstm_24_while_placeholder-
)sequential_23_lstm_24_while_placeholder_1-
)sequential_23_lstm_24_while_placeholder_2-
)sequential_23_lstm_24_while_placeholder_3G
Csequential_23_lstm_24_while_sequential_23_lstm_24_strided_slice_1_0
sequential_23_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_24_tensorarrayunstack_tensorlistfromtensor_0Z
Hsequential_23_lstm_24_while_lstm_cell_24_split_readvariableop_resource_0:X
Jsequential_23_lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0:T
Bsequential_23_lstm_24_while_lstm_cell_24_readvariableop_resource_0:(
$sequential_23_lstm_24_while_identity*
&sequential_23_lstm_24_while_identity_1*
&sequential_23_lstm_24_while_identity_2*
&sequential_23_lstm_24_while_identity_3*
&sequential_23_lstm_24_while_identity_4*
&sequential_23_lstm_24_while_identity_5E
Asequential_23_lstm_24_while_sequential_23_lstm_24_strided_slice_1
}sequential_23_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_24_tensorarrayunstack_tensorlistfromtensorX
Fsequential_23_lstm_24_while_lstm_cell_24_split_readvariableop_resource:V
Hsequential_23_lstm_24_while_lstm_cell_24_split_1_readvariableop_resource:R
@sequential_23_lstm_24_while_lstm_cell_24_readvariableop_resource:¢7sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp¢9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_1¢9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_2¢9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_3¢=sequential_23/lstm_24/while/lstm_cell_24/split/ReadVariableOp¢?sequential_23/lstm_24/while/lstm_cell_24/split_1/ReadVariableOp
Msequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
?sequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemsequential_23_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_24_tensorarrayunstack_tensorlistfromtensor_0'sequential_23_lstm_24_while_placeholderVsequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0®
8sequential_23/lstm_24/while/lstm_cell_24/ones_like/ShapeShapeFsequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0*
T0*
_output_shapes
:}
8sequential_23/lstm_24/while/lstm_cell_24/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ò
2sequential_23/lstm_24/while/lstm_cell_24/ones_likeFillAsequential_23/lstm_24/while/lstm_cell_24/ones_like/Shape:output:0Asequential_23/lstm_24/while/lstm_cell_24/ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
:sequential_23/lstm_24/while/lstm_cell_24/ones_like_1/ShapeShape)sequential_23_lstm_24_while_placeholder_2*
T0*
_output_shapes
:
:sequential_23/lstm_24/while/lstm_cell_24/ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?ø
4sequential_23/lstm_24/while/lstm_cell_24/ones_like_1FillCsequential_23/lstm_24/while/lstm_cell_24/ones_like_1/Shape:output:0Csequential_23/lstm_24/while/lstm_cell_24/ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿê
,sequential_23/lstm_24/while/lstm_cell_24/mulMulFsequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_23/lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
.sequential_23/lstm_24/while/lstm_cell_24/mul_1MulFsequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_23/lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
.sequential_23/lstm_24/while/lstm_cell_24/mul_2MulFsequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_23/lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿì
.sequential_23/lstm_24/while/lstm_cell_24/mul_3MulFsequential_23/lstm_24/while/TensorArrayV2Read/TensorListGetItem:item:0;sequential_23/lstm_24/while/lstm_cell_24/ones_like:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿz
8sequential_23/lstm_24/while/lstm_cell_24/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Æ
=sequential_23/lstm_24/while/lstm_cell_24/split/ReadVariableOpReadVariableOpHsequential_23_lstm_24_while_lstm_cell_24_split_readvariableop_resource_0*
_output_shapes

:*
dtype0
.sequential_23/lstm_24/while/lstm_cell_24/splitSplitAsequential_23/lstm_24/while/lstm_cell_24/split/split_dim:output:0Esequential_23/lstm_24/while/lstm_cell_24/split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_splitÖ
/sequential_23/lstm_24/while/lstm_cell_24/MatMulMatMul0sequential_23/lstm_24/while/lstm_cell_24/mul:z:07sequential_23/lstm_24/while/lstm_cell_24/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
1sequential_23/lstm_24/while/lstm_cell_24/MatMul_1MatMul2sequential_23/lstm_24/while/lstm_cell_24/mul_1:z:07sequential_23/lstm_24/while/lstm_cell_24/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
1sequential_23/lstm_24/while/lstm_cell_24/MatMul_2MatMul2sequential_23/lstm_24/while/lstm_cell_24/mul_2:z:07sequential_23/lstm_24/while/lstm_cell_24/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÚ
1sequential_23/lstm_24/while/lstm_cell_24/MatMul_3MatMul2sequential_23/lstm_24/while/lstm_cell_24/mul_3:z:07sequential_23/lstm_24/while/lstm_cell_24/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ|
:sequential_23/lstm_24/while/lstm_cell_24/split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : Æ
?sequential_23/lstm_24/while/lstm_cell_24/split_1/ReadVariableOpReadVariableOpJsequential_23_lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0*
_output_shapes
:*
dtype0
0sequential_23/lstm_24/while/lstm_cell_24/split_1SplitCsequential_23/lstm_24/while/lstm_cell_24/split_1/split_dim:output:0Gsequential_23/lstm_24/while/lstm_cell_24/split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splitã
0sequential_23/lstm_24/while/lstm_cell_24/BiasAddBiasAdd9sequential_23/lstm_24/while/lstm_cell_24/MatMul:product:09sequential_23/lstm_24/while/lstm_cell_24/split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
2sequential_23/lstm_24/while/lstm_cell_24/BiasAdd_1BiasAdd;sequential_23/lstm_24/while/lstm_cell_24/MatMul_1:product:09sequential_23/lstm_24/while/lstm_cell_24/split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
2sequential_23/lstm_24/while/lstm_cell_24/BiasAdd_2BiasAdd;sequential_23/lstm_24/while/lstm_cell_24/MatMul_2:product:09sequential_23/lstm_24/while/lstm_cell_24/split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿç
2sequential_23/lstm_24/while/lstm_cell_24/BiasAdd_3BiasAdd;sequential_23/lstm_24/while/lstm_cell_24/MatMul_3:product:09sequential_23/lstm_24/while/lstm_cell_24/split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.sequential_23/lstm_24/while/lstm_cell_24/mul_4Mul)sequential_23_lstm_24_while_placeholder_2=sequential_23/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.sequential_23/lstm_24/while/lstm_cell_24/mul_5Mul)sequential_23_lstm_24_while_placeholder_2=sequential_23/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.sequential_23/lstm_24/while/lstm_cell_24/mul_6Mul)sequential_23_lstm_24_while_placeholder_2=sequential_23/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.sequential_23/lstm_24/while/lstm_cell_24/mul_7Mul)sequential_23_lstm_24_while_placeholder_2=sequential_23/lstm_24/while/lstm_cell_24/ones_like_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿº
7sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOpReadVariableOpBsequential_23_lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
<sequential_23/lstm_24/while/lstm_cell_24/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        
>sequential_23/lstm_24/while/lstm_cell_24/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
>sequential_23/lstm_24/while/lstm_cell_24/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      ¸
6sequential_23/lstm_24/while/lstm_cell_24/strided_sliceStridedSlice?sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp:value:0Esequential_23/lstm_24/while/lstm_cell_24/strided_slice/stack:output:0Gsequential_23/lstm_24/while/lstm_cell_24/strided_slice/stack_1:output:0Gsequential_23/lstm_24/while/lstm_cell_24/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskâ
1sequential_23/lstm_24/while/lstm_cell_24/MatMul_4MatMul2sequential_23/lstm_24/while/lstm_cell_24/mul_4:z:0?sequential_23/lstm_24/while/lstm_cell_24/strided_slice:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿß
,sequential_23/lstm_24/while/lstm_cell_24/addAddV29sequential_23/lstm_24/while/lstm_cell_24/BiasAdd:output:0;sequential_23/lstm_24/while/lstm_cell_24/MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
0sequential_23/lstm_24/while/lstm_cell_24/SigmoidSigmoid0sequential_23/lstm_24/while/lstm_cell_24/add:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_1ReadVariableOpBsequential_23_lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
>sequential_23/lstm_24/while/lstm_cell_24/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"       
@sequential_23/lstm_24/while/lstm_cell_24/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
@sequential_23/lstm_24/while/lstm_cell_24/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
8sequential_23/lstm_24/while/lstm_cell_24/strided_slice_1StridedSliceAsequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_1:value:0Gsequential_23/lstm_24/while/lstm_cell_24/strided_slice_1/stack:output:0Isequential_23/lstm_24/while/lstm_cell_24/strided_slice_1/stack_1:output:0Isequential_23/lstm_24/while/lstm_cell_24/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskä
1sequential_23/lstm_24/while/lstm_cell_24/MatMul_5MatMul2sequential_23/lstm_24/while/lstm_cell_24/mul_5:z:0Asequential_23/lstm_24/while/lstm_cell_24/strided_slice_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
.sequential_23/lstm_24/while/lstm_cell_24/add_1AddV2;sequential_23/lstm_24/while/lstm_cell_24/BiasAdd_1:output:0;sequential_23/lstm_24/while/lstm_cell_24/MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
2sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_1Sigmoid2sequential_23/lstm_24/while/lstm_cell_24/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÊ
.sequential_23/lstm_24/while/lstm_cell_24/mul_8Mul6sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_1:y:0)sequential_23_lstm_24_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_2ReadVariableOpBsequential_23_lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
>sequential_23/lstm_24/while/lstm_cell_24/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"       
@sequential_23/lstm_24/while/lstm_cell_24/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"       
@sequential_23/lstm_24/while/lstm_cell_24/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
8sequential_23/lstm_24/while/lstm_cell_24/strided_slice_2StridedSliceAsequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_2:value:0Gsequential_23/lstm_24/while/lstm_cell_24/strided_slice_2/stack:output:0Isequential_23/lstm_24/while/lstm_cell_24/strided_slice_2/stack_1:output:0Isequential_23/lstm_24/while/lstm_cell_24/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskä
1sequential_23/lstm_24/while/lstm_cell_24/MatMul_6MatMul2sequential_23/lstm_24/while/lstm_cell_24/mul_6:z:0Asequential_23/lstm_24/while/lstm_cell_24/strided_slice_2:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
.sequential_23/lstm_24/while/lstm_cell_24/add_2AddV2;sequential_23/lstm_24/while/lstm_cell_24/BiasAdd_2:output:0;sequential_23/lstm_24/while/lstm_cell_24/MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
2sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_2Sigmoid2sequential_23/lstm_24/while/lstm_cell_24/add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÕ
.sequential_23/lstm_24/while/lstm_cell_24/mul_9Mul4sequential_23/lstm_24/while/lstm_cell_24/Sigmoid:y:06sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÑ
.sequential_23/lstm_24/while/lstm_cell_24/add_3AddV22sequential_23/lstm_24/while/lstm_cell_24/mul_8:z:02sequential_23/lstm_24/while/lstm_cell_24/mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¼
9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_3ReadVariableOpBsequential_23_lstm_24_while_lstm_cell_24_readvariableop_resource_0*
_output_shapes

:*
dtype0
>sequential_23/lstm_24/while/lstm_cell_24/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"       
@sequential_23/lstm_24/while/lstm_cell_24/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        
@sequential_23/lstm_24/while/lstm_cell_24/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      Â
8sequential_23/lstm_24/while/lstm_cell_24/strided_slice_3StridedSliceAsequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_3:value:0Gsequential_23/lstm_24/while/lstm_cell_24/strided_slice_3/stack:output:0Isequential_23/lstm_24/while/lstm_cell_24/strided_slice_3/stack_1:output:0Isequential_23/lstm_24/while/lstm_cell_24/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes

:*

begin_mask*
end_maskä
1sequential_23/lstm_24/while/lstm_cell_24/MatMul_7MatMul2sequential_23/lstm_24/while/lstm_cell_24/mul_7:z:0Asequential_23/lstm_24/while/lstm_cell_24/strided_slice_3:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿã
.sequential_23/lstm_24/while/lstm_cell_24/add_4AddV2;sequential_23/lstm_24/while/lstm_cell_24/BiasAdd_3:output:0;sequential_23/lstm_24/while/lstm_cell_24/MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
2sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_3Sigmoid2sequential_23/lstm_24/while/lstm_cell_24/add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ£
2sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_4Sigmoid2sequential_23/lstm_24/while/lstm_cell_24/add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿØ
/sequential_23/lstm_24/while/lstm_cell_24/mul_10Mul6sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_3:y:06sequential_23/lstm_24/while/lstm_cell_24/Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
@sequential_23/lstm_24/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem)sequential_23_lstm_24_while_placeholder_1'sequential_23_lstm_24_while_placeholder3sequential_23/lstm_24/while/lstm_cell_24/mul_10:z:0*
_output_shapes
: *
element_dtype0:éèÒc
!sequential_23/lstm_24/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_23/lstm_24/while/addAddV2'sequential_23_lstm_24_while_placeholder*sequential_23/lstm_24/while/add/y:output:0*
T0*
_output_shapes
: e
#sequential_23/lstm_24/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :¿
!sequential_23/lstm_24/while/add_1AddV2Dsequential_23_lstm_24_while_sequential_23_lstm_24_while_loop_counter,sequential_23/lstm_24/while/add_1/y:output:0*
T0*
_output_shapes
: 
$sequential_23/lstm_24/while/IdentityIdentity%sequential_23/lstm_24/while/add_1:z:0!^sequential_23/lstm_24/while/NoOp*
T0*
_output_shapes
: Â
&sequential_23/lstm_24/while/Identity_1IdentityJsequential_23_lstm_24_while_sequential_23_lstm_24_while_maximum_iterations!^sequential_23/lstm_24/while/NoOp*
T0*
_output_shapes
: 
&sequential_23/lstm_24/while/Identity_2Identity#sequential_23/lstm_24/while/add:z:0!^sequential_23/lstm_24/while/NoOp*
T0*
_output_shapes
: Û
&sequential_23/lstm_24/while/Identity_3IdentityPsequential_23/lstm_24/while/TensorArrayV2Write/TensorListSetItem:output_handle:0!^sequential_23/lstm_24/while/NoOp*
T0*
_output_shapes
: :éèÒ¼
&sequential_23/lstm_24/while/Identity_4Identity3sequential_23/lstm_24/while/lstm_cell_24/mul_10:z:0!^sequential_23/lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ»
&sequential_23/lstm_24/while/Identity_5Identity2sequential_23/lstm_24/while/lstm_cell_24/add_3:z:0!^sequential_23/lstm_24/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÒ
 sequential_23/lstm_24/while/NoOpNoOp8^sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp:^sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_1:^sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_2:^sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_3>^sequential_23/lstm_24/while/lstm_cell_24/split/ReadVariableOp@^sequential_23/lstm_24/while/lstm_cell_24/split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "U
$sequential_23_lstm_24_while_identity-sequential_23/lstm_24/while/Identity:output:0"Y
&sequential_23_lstm_24_while_identity_1/sequential_23/lstm_24/while/Identity_1:output:0"Y
&sequential_23_lstm_24_while_identity_2/sequential_23/lstm_24/while/Identity_2:output:0"Y
&sequential_23_lstm_24_while_identity_3/sequential_23/lstm_24/while/Identity_3:output:0"Y
&sequential_23_lstm_24_while_identity_4/sequential_23/lstm_24/while/Identity_4:output:0"Y
&sequential_23_lstm_24_while_identity_5/sequential_23/lstm_24/while/Identity_5:output:0"
@sequential_23_lstm_24_while_lstm_cell_24_readvariableop_resourceBsequential_23_lstm_24_while_lstm_cell_24_readvariableop_resource_0"
Hsequential_23_lstm_24_while_lstm_cell_24_split_1_readvariableop_resourceJsequential_23_lstm_24_while_lstm_cell_24_split_1_readvariableop_resource_0"
Fsequential_23_lstm_24_while_lstm_cell_24_split_readvariableop_resourceHsequential_23_lstm_24_while_lstm_cell_24_split_readvariableop_resource_0"
Asequential_23_lstm_24_while_sequential_23_lstm_24_strided_slice_1Csequential_23_lstm_24_while_sequential_23_lstm_24_strided_slice_1_0"
}sequential_23_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_24_tensorarrayunstack_tensorlistfromtensorsequential_23_lstm_24_while_tensorarrayv2read_tensorlistgetitem_sequential_23_lstm_24_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : 2r
7sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp7sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp2v
9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_19sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_12v
9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_29sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_22v
9sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_39sequential_23/lstm_24/while/lstm_cell_24/ReadVariableOp_32~
=sequential_23/lstm_24/while/lstm_cell_24/split/ReadVariableOp=sequential_23/lstm_24/while/lstm_cell_24/split/ReadVariableOp2
?sequential_23/lstm_24/while/lstm_cell_24/split_1/ReadVariableOp?sequential_23/lstm_24/while/lstm_cell_24/split_1/ReadVariableOp: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
: 
è
ó
-__inference_lstm_cell_24_layer_call_fn_484754

inputs
states_0
states_1
unknown:
	unknown_0:
	unknown_1:
identity

identity_1

identity_2¢StatefulPartitionedCall¨
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481550o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1
Í~
¨
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_484982

inputs
states_0
states_1/
split_readvariableop_resource:-
split_1_readvariableop_resource:)
readvariableop_resource:
identity

identity_1

identity_2¢ReadVariableOp¢ReadVariableOp_1¢ReadVariableOp_2¢ReadVariableOp_3¢split/ReadVariableOp¢split_1/ReadVariableOpE
ones_like/ShapeShapeinputs*
T0*
_output_shapes
:T
ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?w
	ones_likeFillones_like/Shape:output:0ones_like/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?p
dropout/MulMulones_like:output:0dropout/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿO
dropout/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¦
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_1/MulMulones_like:output:0dropout_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dropout_1/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_1/random_uniform/RandomUniformRandomUniformdropout_1/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_1/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_1/GreaterEqualGreaterEqual/dropout_1/random_uniform/RandomUniform:output:0!dropout_1/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_1/CastCastdropout_1/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_1/Mul_1Muldropout_1/Mul:z:0dropout_1/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_2/MulMulones_like:output:0dropout_2/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dropout_2/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_2/random_uniform/RandomUniformRandomUniformdropout_2/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_2/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_2/GreaterEqualGreaterEqual/dropout_2/random_uniform/RandomUniform:output:0!dropout_2/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_2/CastCastdropout_2/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_2/Mul_1Muldropout_2/Mul:z:0dropout_2/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?t
dropout_3/MulMulones_like:output:0dropout_3/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
dropout_3/ShapeShapeones_like:output:0*
T0*
_output_shapes
:
&dropout_3/random_uniform/RandomUniformRandomUniformdropout_3/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_3/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_3/GreaterEqualGreaterEqual/dropout_3/random_uniform/RandomUniform:output:0!dropout_3/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_3/CastCastdropout_3/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_3/Mul_1Muldropout_3/Mul:z:0dropout_3/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿI
ones_like_1/ShapeShapestates_0*
T0*
_output_shapes
:V
ones_like_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?}
ones_like_1Fillones_like_1/Shape:output:0ones_like_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_4/MulMulones_like_1:output:0dropout_4/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_4/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_4/random_uniform/RandomUniformRandomUniformdropout_4/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_4/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_4/GreaterEqualGreaterEqual/dropout_4/random_uniform/RandomUniform:output:0!dropout_4/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_4/CastCastdropout_4/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_4/Mul_1Muldropout_4/Mul:z:0dropout_4/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_5/MulMulones_like_1:output:0dropout_5/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_5/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_5/random_uniform/RandomUniformRandomUniformdropout_5/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_5/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_5/GreaterEqualGreaterEqual/dropout_5/random_uniform/RandomUniform:output:0!dropout_5/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_5/CastCastdropout_5/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_5/Mul_1Muldropout_5/Mul:z:0dropout_5/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_6/MulMulones_like_1:output:0dropout_6/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_6/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_6/random_uniform/RandomUniformRandomUniformdropout_6/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_6/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_6/GreaterEqualGreaterEqual/dropout_6/random_uniform/RandomUniform:output:0!dropout_6/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_6/CastCastdropout_6/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_6/Mul_1Muldropout_6/Mul:z:0dropout_6/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿT
dropout_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *ä8?v
dropout_7/MulMulones_like_1:output:0dropout_7/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
dropout_7/ShapeShapeones_like_1:output:0*
T0*
_output_shapes
:
&dropout_7/random_uniform/RandomUniformRandomUniformdropout_7/Shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0]
dropout_7/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *ÍÌÌ=¬
dropout_7/GreaterEqualGreaterEqual/dropout_7/random_uniform/RandomUniform:output:0!dropout_7/GreaterEqual/y:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
dropout_7/CastCastdropout_7/GreaterEqual:z:0*

DstT0*

SrcT0
*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿo
dropout_7/Mul_1Muldropout_7/Mul:z:0dropout_7/Cast:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
mulMulinputsdropout/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_1Mulinputsdropout_1/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_2Mulinputsdropout_2/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[
mul_3Mulinputsdropout_3/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :r
split/ReadVariableOpReadVariableOpsplit_readvariableop_resource*
_output_shapes

:*
dtype0
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*<
_output_shapes*
(::::*
	num_split[
MatMulMatMulmul:z:0split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_1MatMul	mul_1:z:0split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_2MatMul	mul_2:z:0split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
MatMul_3MatMul	mul_3:z:0split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿS
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : r
split_1/ReadVariableOpReadVariableOpsplit_1_readvariableop_resource*
_output_shapes
:*
dtype0
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*,
_output_shapes
::::*
	num_splith
BiasAddBiasAddMatMul:product:0split_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_1BiasAddMatMul_1:product:0split_1:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_2BiasAddMatMul_2:product:0split_1:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
	BiasAdd_3BiasAddMatMul_3:product:0split_1:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_4Mulstates_0dropout_4/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_5Mulstates_0dropout_5/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_6Mulstates_0dropout_6/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_7Mulstates_0dropout_7/Mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿf
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
valueB"      ë
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
:ÿÿÿÿÿÿÿÿÿd
addAddV2BiasAdd:output:0MatMul_4:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿM
SigmoidSigmoidadd:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_1AddV2BiasAdd_1:output:0MatMul_5:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_1Sigmoid	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿW
mul_8MulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_2AddV2BiasAdd_2:output:0MatMul_6:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_2Sigmoid	add_2:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ
mul_9MulSigmoid:y:0Sigmoid_2:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
add_3AddV2	mul_8:z:0	mul_9:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
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
valueB"      õ
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
:ÿÿÿÿÿÿÿÿÿh
add_4AddV2BiasAdd_3:output:0MatMul_7:product:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_3Sigmoid	add_4:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
	Sigmoid_4Sigmoid	add_3:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
mul_10MulSigmoid_3:y:0Sigmoid_4:y:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ[

Identity_1Identity
mul_10:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿZ

Identity_2Identity	add_3:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÀ
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^ReadVariableOp_2^ReadVariableOp_3^split/ReadVariableOp^split_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22$
ReadVariableOp_3ReadVariableOp_32,
split/ReadVariableOpsplit/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
states/1

É
"__inference__traced_restore_485231
file_prefix2
 assignvariableop_dense_72_kernel:.
 assignvariableop_1_dense_72_bias:4
"assignvariableop_2_dense_73_kernel:.
 assignvariableop_3_dense_73_bias:4
"assignvariableop_4_dense_74_kernel:.
 assignvariableop_5_dense_74_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: A
/assignvariableop_11_lstm_24_lstm_cell_24_kernel:K
9assignvariableop_12_lstm_24_lstm_cell_24_recurrent_kernel:;
-assignvariableop_13_lstm_24_lstm_cell_24_bias:#
assignvariableop_14_total: #
assignvariableop_15_count: %
assignvariableop_16_total_1: %
assignvariableop_17_count_1: <
*assignvariableop_18_adam_dense_72_kernel_m:6
(assignvariableop_19_adam_dense_72_bias_m:<
*assignvariableop_20_adam_dense_73_kernel_m:6
(assignvariableop_21_adam_dense_73_bias_m:<
*assignvariableop_22_adam_dense_74_kernel_m:6
(assignvariableop_23_adam_dense_74_bias_m:H
6assignvariableop_24_adam_lstm_24_lstm_cell_24_kernel_m:R
@assignvariableop_25_adam_lstm_24_lstm_cell_24_recurrent_kernel_m:B
4assignvariableop_26_adam_lstm_24_lstm_cell_24_bias_m:<
*assignvariableop_27_adam_dense_72_kernel_v:6
(assignvariableop_28_adam_dense_72_bias_v:<
*assignvariableop_29_adam_dense_73_kernel_v:6
(assignvariableop_30_adam_dense_73_bias_v:<
*assignvariableop_31_adam_dense_74_kernel_v:6
(assignvariableop_32_adam_dense_74_bias_v:H
6assignvariableop_33_adam_lstm_24_lstm_cell_24_kernel_v:R
@assignvariableop_34_adam_lstm_24_lstm_cell_24_recurrent_kernel_v:B
4assignvariableop_35_adam_lstm_24_lstm_cell_24_bias_v:
identity_37¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*®
value¤B¡%B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHº
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:%*
dtype0*]
valueTBR%B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ú
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ª
_output_shapes
:::::::::::::::::::::::::::::::::::::*3
dtypes)
'2%	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp assignvariableop_dense_72_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp assignvariableop_1_dense_72_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_dense_73_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_dense_73_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_dense_74_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_dense_74_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_11AssignVariableOp/assignvariableop_11_lstm_24_lstm_cell_24_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:ª
AssignVariableOp_12AssignVariableOp9assignvariableop_12_lstm_24_lstm_cell_24_recurrent_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp-assignvariableop_13_lstm_24_lstm_cell_24_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_totalIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOpassignvariableop_15_countIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOpassignvariableop_16_total_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOpassignvariableop_17_count_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_72_kernel_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp(assignvariableop_19_adam_dense_72_bias_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_73_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp(assignvariableop_21_adam_dense_73_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_74_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp(assignvariableop_23_adam_dense_74_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_24AssignVariableOp6assignvariableop_24_adam_lstm_24_lstm_cell_24_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_25AssignVariableOp@assignvariableop_25_adam_lstm_24_lstm_cell_24_recurrent_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_26AssignVariableOp4assignvariableop_26_adam_lstm_24_lstm_cell_24_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOp*assignvariableop_27_adam_dense_72_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_dense_72_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOp*assignvariableop_29_adam_dense_73_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_dense_73_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOp*assignvariableop_31_adam_dense_74_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOp(assignvariableop_32_adam_dense_74_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_33AssignVariableOp6assignvariableop_33_adam_lstm_24_lstm_cell_24_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:±
AssignVariableOp_34AssignVariableOp@assignvariableop_34_adam_lstm_24_lstm_cell_24_recurrent_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_35AssignVariableOp4assignvariableop_35_adam_lstm_24_lstm_cell_24_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ç
Identity_36Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_37IdentityIdentity_36:output:0^NoOp_1*
T0*
_output_shapes
: Ô
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
8

C__inference_lstm_24_layer_call_and_return_conditional_losses_481373

inputs%
lstm_cell_24_481291:!
lstm_cell_24_481293:%
lstm_cell_24_481295:
identity¢$lstm_cell_24/StatefulPartitionedCall¢while;
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
valueB:Ñ
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
:ÿÿÿÿÿÿÿÿÿR
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
:ÿÿÿÿÿÿÿÿÿc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
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
valueB:Û
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
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
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
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskõ
$lstm_cell_24/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_24_481291lstm_cell_24_481293lstm_cell_24_481295*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ*%
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_481290n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¸
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
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
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ·
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_24_481291lstm_cell_24_481293lstm_cell_24_481295*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_481304*
condR
while_cond_481303*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   Ë
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
element_dtype0h
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿu
NoOpNoOp%^lstm_cell_24/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2L
$lstm_cell_24/StatefulPartitionedCall$lstm_cell_24/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_484483
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_484483___redundant_placeholder04
0while_while_cond_484483___redundant_placeholder14
0while_while_cond_484483___redundant_placeholder24
0while_while_cond_484483___redundant_placeholder3
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
@: : : : :ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ: ::::: 
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
:ÿÿÿÿÿÿÿÿÿ:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ:

_output_shapes
: :

_output_shapes
:


â
.__inference_sequential_23_layer_call_fn_482034
dense_72_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:
	unknown_7:
identity¢StatefulPartitionedCallÁ
StatefulPartitionedCallStatefulPartitionedCalldense_72_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7*
Tin
2
*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*+
_read_only_resource_inputs
		*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_sequential_23_layer_call_and_return_conditional_losses_482013o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*<
_input_shapes+
):ÿÿÿÿÿÿÿÿÿ: : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(
_user_specified_namedense_72_input"ÛL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*½
serving_default©
M
dense_72_input;
 serving_default_dense_72_input:0ÿÿÿÿÿÿÿÿÿ<
dense_740
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:

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
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
Ú
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
»

kernel
 bias
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses"
_tf_keras_layer
»

'kernel
(bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
-__call__
*.&call_and_return_all_conditional_losses"
_tf_keras_layer
õ
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
Ê
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
2
.__inference_sequential_23_layer_call_fn_482034
.__inference_sequential_23_layer_call_fn_482633
.__inference_sequential_23_layer_call_fn_482656
.__inference_sequential_23_layer_call_fn_482552À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ò2ï
I__inference_sequential_23_layer_call_and_return_conditional_losses_482937
I__inference_sequential_23_layer_call_and_return_conditional_losses_483346
I__inference_sequential_23_layer_call_and_return_conditional_losses_482578
I__inference_sequential_23_layer_call_and_return_conditional_losses_482604À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÓBÐ
!__inference__wrapped_model_481173dense_72_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
,
<serving_default"
signature_map
!:2dense_72/kernel
:2dense_72/bias
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
­
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
Ó2Ð
)__inference_dense_72_layer_call_fn_483380¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_72_layer_call_and_return_conditional_losses_483410¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ø
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
¹

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
2
(__inference_lstm_24_layer_call_fn_483421
(__inference_lstm_24_layer_call_fn_483432
(__inference_lstm_24_layer_call_fn_483443
(__inference_lstm_24_layer_call_fn_483454Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ï2ì
C__inference_lstm_24_layer_call_and_return_conditional_losses_483697
C__inference_lstm_24_layer_call_and_return_conditional_losses_484068
C__inference_lstm_24_layer_call_and_return_conditional_losses_484311
C__inference_lstm_24_layer_call_and_return_conditional_losses_484682Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
!:2dense_73/kernel
:2dense_73/bias
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
­
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
Ó2Ð
)__inference_dense_73_layer_call_fn_484691¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_73_layer_call_and_return_conditional_losses_484701¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
!:2dense_74/kernel
:2dense_74/bias
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
­
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
Ó2Ð
)__inference_dense_74_layer_call_fn_484710¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
î2ë
D__inference_dense_74_layer_call_and_return_conditional_losses_484720¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+2lstm_24/lstm_cell_24/kernel
7:52%lstm_24/lstm_cell_24/recurrent_kernel
':%2lstm_24/lstm_cell_24/bias
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
ÒBÏ
$__inference_signature_wrapper_483371dense_72_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
­
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
¢2
-__inference_lstm_cell_24_layer_call_fn_484737
-__inference_lstm_cell_24_layer_call_fn_484754¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
Ø2Õ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_484836
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_484982¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
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
&:$2Adam/dense_72/kernel/m
 :2Adam/dense_72/bias/m
&:$2Adam/dense_73/kernel/m
 :2Adam/dense_73/bias/m
&:$2Adam/dense_74/kernel/m
 :2Adam/dense_74/bias/m
2:02"Adam/lstm_24/lstm_cell_24/kernel/m
<::2,Adam/lstm_24/lstm_cell_24/recurrent_kernel/m
,:*2 Adam/lstm_24/lstm_cell_24/bias/m
&:$2Adam/dense_72/kernel/v
 :2Adam/dense_72/bias/v
&:$2Adam/dense_73/kernel/v
 :2Adam/dense_73/bias/v
&:$2Adam/dense_74/kernel/v
 :2Adam/dense_74/bias/v
2:02"Adam/lstm_24/lstm_cell_24/kernel/v
<::2,Adam/lstm_24/lstm_cell_24/recurrent_kernel/v
,:*2 Adam/lstm_24/lstm_cell_24/bias/v¢
!__inference__wrapped_model_481173}	465 '(;¢8
1¢.
,)
dense_72_inputÿÿÿÿÿÿÿÿÿ
ª "3ª0
.
dense_74"
dense_74ÿÿÿÿÿÿÿÿÿ¬
D__inference_dense_72_layer_call_and_return_conditional_losses_483410d3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
)__inference_dense_72_layer_call_fn_483380W3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_73_layer_call_and_return_conditional_losses_484701\ /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_73_layer_call_fn_484691O /¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
D__inference_dense_74_layer_call_and_return_conditional_losses_484720\'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 |
)__inference_dense_74_layer_call_fn_484710O'(/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿÄ
C__inference_lstm_24_layer_call_and_return_conditional_losses_483697}465O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
C__inference_lstm_24_layer_call_and_return_conditional_losses_484068}465O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
C__inference_lstm_24_layer_call_and_return_conditional_losses_484311m465?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ´
C__inference_lstm_24_layer_call_and_return_conditional_losses_484682m465?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
(__inference_lstm_24_layer_call_fn_483421p465O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_24_layer_call_fn_483432p465O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_24_layer_call_fn_483443`465?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
(__inference_lstm_24_layer_call_fn_483454`465?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿÊ
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_484836ý465¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 Ê
H__inference_lstm_cell_24_layer_call_and_return_conditional_losses_484982ý465¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿ
EB

0/1/0ÿÿÿÿÿÿÿÿÿ

0/1/1ÿÿÿÿÿÿÿÿÿ
 
-__inference_lstm_cell_24_layer_call_fn_484737í465¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿ
-__inference_lstm_cell_24_layer_call_fn_484754í465¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿ
"
states/1ÿÿÿÿÿÿÿÿÿ
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿ
A>

1/0ÿÿÿÿÿÿÿÿÿ

1/1ÿÿÿÿÿÿÿÿÿÄ
I__inference_sequential_23_layer_call_and_return_conditional_losses_482578w	465 '(C¢@
9¢6
,)
dense_72_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ä
I__inference_sequential_23_layer_call_and_return_conditional_losses_482604w	465 '(C¢@
9¢6
,)
dense_72_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
I__inference_sequential_23_layer_call_and_return_conditional_losses_482937o	465 '(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¼
I__inference_sequential_23_layer_call_and_return_conditional_losses_483346o	465 '(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
.__inference_sequential_23_layer_call_fn_482034j	465 '(C¢@
9¢6
,)
dense_72_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_23_layer_call_fn_482552j	465 '(C¢@
9¢6
,)
dense_72_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_23_layer_call_fn_482633b	465 '(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
.__inference_sequential_23_layer_call_fn_482656b	465 '(;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ¸
$__inference_signature_wrapper_483371	465 '(M¢J
¢ 
Cª@
>
dense_72_input,)
dense_72_inputÿÿÿÿÿÿÿÿÿ"3ª0
.
dense_74"
dense_74ÿÿÿÿÿÿÿÿÿ