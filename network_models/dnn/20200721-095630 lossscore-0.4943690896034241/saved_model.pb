Ё┌
ф§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
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
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.2.02v2.2.0-rc4-8-g2b96f3662b8щ┘
Ћ
sequential_4/dense_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ъ*-
shared_namesequential_4/dense_20/kernel
ј
0sequential_4/dense_20/kernel/Read/ReadVariableOpReadVariableOpsequential_4/dense_20/kernel*
_output_shapes
:	ъ*
dtype0
Ї
sequential_4/dense_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:ъ*+
shared_namesequential_4/dense_20/bias
є
.sequential_4/dense_20/bias/Read/ReadVariableOpReadVariableOpsequential_4/dense_20/bias*
_output_shapes	
:ъ*
dtype0
ќ
sequential_4/dense_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ъе*-
shared_namesequential_4/dense_21/kernel
Ј
0sequential_4/dense_21/kernel/Read/ReadVariableOpReadVariableOpsequential_4/dense_21/kernel* 
_output_shapes
:
ъе*
dtype0
Ї
sequential_4/dense_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:е*+
shared_namesequential_4/dense_21/bias
є
.sequential_4/dense_21/bias/Read/ReadVariableOpReadVariableOpsequential_4/dense_21/bias*
_output_shapes	
:е*
dtype0
ќ
sequential_4/dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
е▓*-
shared_namesequential_4/dense_22/kernel
Ј
0sequential_4/dense_22/kernel/Read/ReadVariableOpReadVariableOpsequential_4/dense_22/kernel* 
_output_shapes
:
е▓*
dtype0
Ї
sequential_4/dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:▓*+
shared_namesequential_4/dense_22/bias
є
.sequential_4/dense_22/bias/Read/ReadVariableOpReadVariableOpsequential_4/dense_22/bias*
_output_shapes	
:▓*
dtype0
ќ
sequential_4/dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
▓╝*-
shared_namesequential_4/dense_23/kernel
Ј
0sequential_4/dense_23/kernel/Read/ReadVariableOpReadVariableOpsequential_4/dense_23/kernel* 
_output_shapes
:
▓╝*
dtype0
Ї
sequential_4/dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:╝*+
shared_namesequential_4/dense_23/bias
є
.sequential_4/dense_23/bias/Read/ReadVariableOpReadVariableOpsequential_4/dense_23/bias*
_output_shapes	
:╝*
dtype0
Ћ
sequential_4/dense_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╝*-
shared_namesequential_4/dense_24/kernel
ј
0sequential_4/dense_24/kernel/Read/ReadVariableOpReadVariableOpsequential_4/dense_24/kernel*
_output_shapes
:	╝*
dtype0
ї
sequential_4/dense_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_namesequential_4/dense_24/bias
Ё
.sequential_4/dense_24/bias/Read/ReadVariableOpReadVariableOpsequential_4/dense_24/bias*
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
Б
#Adam/sequential_4/dense_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ъ*4
shared_name%#Adam/sequential_4/dense_20/kernel/m
ю
7Adam/sequential_4/dense_20/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_20/kernel/m*
_output_shapes
:	ъ*
dtype0
Џ
!Adam/sequential_4/dense_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:ъ*2
shared_name#!Adam/sequential_4/dense_20/bias/m
ћ
5Adam/sequential_4/dense_20/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_20/bias/m*
_output_shapes	
:ъ*
dtype0
ц
#Adam/sequential_4/dense_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ъе*4
shared_name%#Adam/sequential_4/dense_21/kernel/m
Ю
7Adam/sequential_4/dense_21/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_21/kernel/m* 
_output_shapes
:
ъе*
dtype0
Џ
!Adam/sequential_4/dense_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:е*2
shared_name#!Adam/sequential_4/dense_21/bias/m
ћ
5Adam/sequential_4/dense_21/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_21/bias/m*
_output_shapes	
:е*
dtype0
ц
#Adam/sequential_4/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
е▓*4
shared_name%#Adam/sequential_4/dense_22/kernel/m
Ю
7Adam/sequential_4/dense_22/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_22/kernel/m* 
_output_shapes
:
е▓*
dtype0
Џ
!Adam/sequential_4/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:▓*2
shared_name#!Adam/sequential_4/dense_22/bias/m
ћ
5Adam/sequential_4/dense_22/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_22/bias/m*
_output_shapes	
:▓*
dtype0
ц
#Adam/sequential_4/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
▓╝*4
shared_name%#Adam/sequential_4/dense_23/kernel/m
Ю
7Adam/sequential_4/dense_23/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_23/kernel/m* 
_output_shapes
:
▓╝*
dtype0
Џ
!Adam/sequential_4/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:╝*2
shared_name#!Adam/sequential_4/dense_23/bias/m
ћ
5Adam/sequential_4/dense_23/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_23/bias/m*
_output_shapes	
:╝*
dtype0
Б
#Adam/sequential_4/dense_24/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╝*4
shared_name%#Adam/sequential_4/dense_24/kernel/m
ю
7Adam/sequential_4/dense_24/kernel/m/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_24/kernel/m*
_output_shapes
:	╝*
dtype0
џ
!Adam/sequential_4/dense_24/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_4/dense_24/bias/m
Њ
5Adam/sequential_4/dense_24/bias/m/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_24/bias/m*
_output_shapes
:*
dtype0
Б
#Adam/sequential_4/dense_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	ъ*4
shared_name%#Adam/sequential_4/dense_20/kernel/v
ю
7Adam/sequential_4/dense_20/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_20/kernel/v*
_output_shapes
:	ъ*
dtype0
Џ
!Adam/sequential_4/dense_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:ъ*2
shared_name#!Adam/sequential_4/dense_20/bias/v
ћ
5Adam/sequential_4/dense_20/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_20/bias/v*
_output_shapes	
:ъ*
dtype0
ц
#Adam/sequential_4/dense_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ъе*4
shared_name%#Adam/sequential_4/dense_21/kernel/v
Ю
7Adam/sequential_4/dense_21/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_21/kernel/v* 
_output_shapes
:
ъе*
dtype0
Џ
!Adam/sequential_4/dense_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:е*2
shared_name#!Adam/sequential_4/dense_21/bias/v
ћ
5Adam/sequential_4/dense_21/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_21/bias/v*
_output_shapes	
:е*
dtype0
ц
#Adam/sequential_4/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
е▓*4
shared_name%#Adam/sequential_4/dense_22/kernel/v
Ю
7Adam/sequential_4/dense_22/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_22/kernel/v* 
_output_shapes
:
е▓*
dtype0
Џ
!Adam/sequential_4/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:▓*2
shared_name#!Adam/sequential_4/dense_22/bias/v
ћ
5Adam/sequential_4/dense_22/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_22/bias/v*
_output_shapes	
:▓*
dtype0
ц
#Adam/sequential_4/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
▓╝*4
shared_name%#Adam/sequential_4/dense_23/kernel/v
Ю
7Adam/sequential_4/dense_23/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_23/kernel/v* 
_output_shapes
:
▓╝*
dtype0
Џ
!Adam/sequential_4/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:╝*2
shared_name#!Adam/sequential_4/dense_23/bias/v
ћ
5Adam/sequential_4/dense_23/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_23/bias/v*
_output_shapes	
:╝*
dtype0
Б
#Adam/sequential_4/dense_24/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	╝*4
shared_name%#Adam/sequential_4/dense_24/kernel/v
ю
7Adam/sequential_4/dense_24/kernel/v/Read/ReadVariableOpReadVariableOp#Adam/sequential_4/dense_24/kernel/v*
_output_shapes
:	╝*
dtype0
џ
!Adam/sequential_4/dense_24/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*2
shared_name#!Adam/sequential_4/dense_24/bias/v
Њ
5Adam/sequential_4/dense_24/bias/v/Read/ReadVariableOpReadVariableOp!Adam/sequential_4/dense_24/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
┴F
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЧE
valueЫEB№E BУE
Ї
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
_build_input_shape
	variables
trainable_variables
regularization_losses
	keras_api

signatures
x
_feature_columns

_resources
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
R
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
R
(	variables
)trainable_variables
*regularization_losses
+	keras_api
h

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
R
2	variables
3trainable_variables
4regularization_losses
5	keras_api
h

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
R
<	variables
=trainable_variables
>regularization_losses
?	keras_api
h

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
ѕ
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratemЇmј"mЈ#mљ,mЉ-mњ6mЊ7mћ@mЋAmќvЌvў"vЎ#vџ,vЏ-vю6vЮ7vъ@vЪAvа
 
F
0
1
"2
#3
,4
-5
66
77
@8
A9
F
0
1
"2
#3
,4
-5
66
77
@8
A9
 
Г
	variables
trainable_variables
Klayer_metrics
Llayer_regularization_losses
Mmetrics

Nlayers
regularization_losses
Onon_trainable_variables
 
 
 
 
 
 
Г
	variables
trainable_variables
Player_metrics
Qlayer_regularization_losses
Rmetrics

Slayers
regularization_losses
Tnon_trainable_variables
hf
VARIABLE_VALUEsequential_4/dense_20/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_4/dense_20/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Г
	variables
trainable_variables
Ulayer_metrics
Vlayer_regularization_losses
Wmetrics

Xlayers
regularization_losses
Ynon_trainable_variables
 
 
 
Г
	variables
trainable_variables
Zlayer_metrics
[layer_regularization_losses
\metrics

]layers
 regularization_losses
^non_trainable_variables
hf
VARIABLE_VALUEsequential_4/dense_21/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_4/dense_21/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
Г
$	variables
%trainable_variables
_layer_metrics
`layer_regularization_losses
ametrics

blayers
&regularization_losses
cnon_trainable_variables
 
 
 
Г
(	variables
)trainable_variables
dlayer_metrics
elayer_regularization_losses
fmetrics

glayers
*regularization_losses
hnon_trainable_variables
hf
VARIABLE_VALUEsequential_4/dense_22/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_4/dense_22/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

,0
-1

,0
-1
 
Г
.	variables
/trainable_variables
ilayer_metrics
jlayer_regularization_losses
kmetrics

llayers
0regularization_losses
mnon_trainable_variables
 
 
 
Г
2	variables
3trainable_variables
nlayer_metrics
olayer_regularization_losses
pmetrics

qlayers
4regularization_losses
rnon_trainable_variables
hf
VARIABLE_VALUEsequential_4/dense_23/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_4/dense_23/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

60
71

60
71
 
Г
8	variables
9trainable_variables
slayer_metrics
tlayer_regularization_losses
umetrics

vlayers
:regularization_losses
wnon_trainable_variables
 
 
 
Г
<	variables
=trainable_variables
xlayer_metrics
ylayer_regularization_losses
zmetrics

{layers
>regularization_losses
|non_trainable_variables
hf
VARIABLE_VALUEsequential_4/dense_24/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEsequential_4/dense_24/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

@0
A1
 
»
B	variables
Ctrainable_variables
}layer_metrics
~layer_regularization_losses
metrics
ђlayers
Dregularization_losses
Ђnon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

ѓ0
Ѓ1
F
0
1
2
3
4
5
6
7
	8

9
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ёtotal

Ёcount
є	variables
Є	keras_api
I

ѕtotal

Ѕcount
і
_fn_kwargs
І	variables
ї	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ё0
Ё1

є	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

ѕ0
Ѕ1

І	variables
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_20/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_20/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_21/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_21/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_22/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_22/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_23/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_23/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_24/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_24/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_20/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_20/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_21/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_21/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_22/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_22/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_23/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_23/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
їЅ
VARIABLE_VALUE#Adam/sequential_4/dense_24/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ѕЁ
VARIABLE_VALUE!Adam/sequential_4/dense_24/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
v
serving_default_AgePlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
{
serving_default_EmbarkedPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
w
serving_default_FarePlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
z
serving_default_IsAlonePlaceholder*'
_output_shapes
:         *
dtype0	*
shape:         
}
serving_default_NameLengthPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
y
serving_default_PclassPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
v
serving_default_SexPlaceholder*'
_output_shapes
:         *
dtype0	*
shape:         
x
serving_default_TitlePlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
ш
StatefulPartitionedCallStatefulPartitionedCallserving_default_Ageserving_default_Embarkedserving_default_Fareserving_default_IsAloneserving_default_NameLengthserving_default_Pclassserving_default_Sexserving_default_Titlesequential_4/dense_20/kernelsequential_4/dense_20/biassequential_4/dense_21/kernelsequential_4/dense_21/biassequential_4/dense_22/kernelsequential_4/dense_22/biassequential_4/dense_23/kernelsequential_4/dense_23/biassequential_4/dense_24/kernelsequential_4/dense_24/bias*
Tin
2		*
Tout
2*'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference_signature_wrapper_213121
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
є
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename0sequential_4/dense_20/kernel/Read/ReadVariableOp.sequential_4/dense_20/bias/Read/ReadVariableOp0sequential_4/dense_21/kernel/Read/ReadVariableOp.sequential_4/dense_21/bias/Read/ReadVariableOp0sequential_4/dense_22/kernel/Read/ReadVariableOp.sequential_4/dense_22/bias/Read/ReadVariableOp0sequential_4/dense_23/kernel/Read/ReadVariableOp.sequential_4/dense_23/bias/Read/ReadVariableOp0sequential_4/dense_24/kernel/Read/ReadVariableOp.sequential_4/dense_24/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp7Adam/sequential_4/dense_20/kernel/m/Read/ReadVariableOp5Adam/sequential_4/dense_20/bias/m/Read/ReadVariableOp7Adam/sequential_4/dense_21/kernel/m/Read/ReadVariableOp5Adam/sequential_4/dense_21/bias/m/Read/ReadVariableOp7Adam/sequential_4/dense_22/kernel/m/Read/ReadVariableOp5Adam/sequential_4/dense_22/bias/m/Read/ReadVariableOp7Adam/sequential_4/dense_23/kernel/m/Read/ReadVariableOp5Adam/sequential_4/dense_23/bias/m/Read/ReadVariableOp7Adam/sequential_4/dense_24/kernel/m/Read/ReadVariableOp5Adam/sequential_4/dense_24/bias/m/Read/ReadVariableOp7Adam/sequential_4/dense_20/kernel/v/Read/ReadVariableOp5Adam/sequential_4/dense_20/bias/v/Read/ReadVariableOp7Adam/sequential_4/dense_21/kernel/v/Read/ReadVariableOp5Adam/sequential_4/dense_21/bias/v/Read/ReadVariableOp7Adam/sequential_4/dense_22/kernel/v/Read/ReadVariableOp5Adam/sequential_4/dense_22/bias/v/Read/ReadVariableOp7Adam/sequential_4/dense_23/kernel/v/Read/ReadVariableOp5Adam/sequential_4/dense_23/bias/v/Read/ReadVariableOp7Adam/sequential_4/dense_24/kernel/v/Read/ReadVariableOp5Adam/sequential_4/dense_24/bias/v/Read/ReadVariableOpConst*4
Tin-
+2)	*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*(
f#R!
__inference__traced_save_213941
ш

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential_4/dense_20/kernelsequential_4/dense_20/biassequential_4/dense_21/kernelsequential_4/dense_21/biassequential_4/dense_22/kernelsequential_4/dense_22/biassequential_4/dense_23/kernelsequential_4/dense_23/biassequential_4/dense_24/kernelsequential_4/dense_24/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1#Adam/sequential_4/dense_20/kernel/m!Adam/sequential_4/dense_20/bias/m#Adam/sequential_4/dense_21/kernel/m!Adam/sequential_4/dense_21/bias/m#Adam/sequential_4/dense_22/kernel/m!Adam/sequential_4/dense_22/bias/m#Adam/sequential_4/dense_23/kernel/m!Adam/sequential_4/dense_23/bias/m#Adam/sequential_4/dense_24/kernel/m!Adam/sequential_4/dense_24/bias/m#Adam/sequential_4/dense_20/kernel/v!Adam/sequential_4/dense_20/bias/v#Adam/sequential_4/dense_21/kernel/v!Adam/sequential_4/dense_21/bias/v#Adam/sequential_4/dense_22/kernel/v!Adam/sequential_4/dense_22/bias/v#Adam/sequential_4/dense_23/kernel/v!Adam/sequential_4/dense_23/bias/v#Adam/sequential_4/dense_24/kernel/v!Adam/sequential_4/dense_24/bias/v*3
Tin,
*2(*
Tout
2*
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*+
f&R$
"__inference__traced_restore_214070╔њ
јX
┌
L__inference_dense_features_4_layer_call_and_return_conditional_losses_212592
features

features_1

features_2

features_3	

features_4

features_5

features_6	

features_7
identityN
	Age/ShapeShapefeatures*
T0*
_output_shapes
:2
	Age/Shape|
Age/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Age/strided_slice/stackђ
Age/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Age/strided_slice/stack_1ђ
Age/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Age/strided_slice/stack_2Щ
Age/strided_sliceStridedSliceAge/Shape:output:0 Age/strided_slice/stack:output:0"Age/strided_slice/stack_1:output:0"Age/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Age/strided_slicel
Age/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Age/Reshape/shape/1ќ
Age/Reshape/shapePackAge/strided_slice:output:0Age/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Age/Reshape/shape}
Age/ReshapeReshapefeaturesAge/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Age/ReshapeZ
Embarked/ShapeShape
features_1*
T0*
_output_shapes
:2
Embarked/Shapeє
Embarked/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Embarked/strided_slice/stackі
Embarked/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Embarked/strided_slice/stack_1і
Embarked/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Embarked/strided_slice/stack_2ў
Embarked/strided_sliceStridedSliceEmbarked/Shape:output:0%Embarked/strided_slice/stack:output:0'Embarked/strided_slice/stack_1:output:0'Embarked/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Embarked/strided_slicev
Embarked/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Embarked/Reshape/shape/1ф
Embarked/Reshape/shapePackEmbarked/strided_slice:output:0!Embarked/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Embarked/Reshape/shapeј
Embarked/ReshapeReshape
features_1Embarked/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Embarked/ReshapeR

Fare/ShapeShape
features_2*
T0*
_output_shapes
:2

Fare/Shape~
Fare/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Fare/strided_slice/stackѓ
Fare/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Fare/strided_slice/stack_1ѓ
Fare/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Fare/strided_slice/stack_2ђ
Fare/strided_sliceStridedSliceFare/Shape:output:0!Fare/strided_slice/stack:output:0#Fare/strided_slice/stack_1:output:0#Fare/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Fare/strided_slicen
Fare/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Fare/Reshape/shape/1џ
Fare/Reshape/shapePackFare/strided_slice:output:0Fare/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Fare/Reshape/shapeѓ
Fare/ReshapeReshape
features_2Fare/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Fare/Reshapeq
IsAlone/CastCast
features_3*

DstT0*

SrcT0	*'
_output_shapes
:         2
IsAlone/Cast^
IsAlone/ShapeShapeIsAlone/Cast:y:0*
T0*
_output_shapes
:2
IsAlone/Shapeё
IsAlone/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
IsAlone/strided_slice/stackѕ
IsAlone/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
IsAlone/strided_slice/stack_1ѕ
IsAlone/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
IsAlone/strided_slice/stack_2њ
IsAlone/strided_sliceStridedSliceIsAlone/Shape:output:0$IsAlone/strided_slice/stack:output:0&IsAlone/strided_slice/stack_1:output:0&IsAlone/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
IsAlone/strided_slicet
IsAlone/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
IsAlone/Reshape/shape/1д
IsAlone/Reshape/shapePackIsAlone/strided_slice:output:0 IsAlone/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
IsAlone/Reshape/shapeЉ
IsAlone/ReshapeReshapeIsAlone/Cast:y:0IsAlone/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
IsAlone/Reshape^
NameLength/ShapeShape
features_4*
T0*
_output_shapes
:2
NameLength/Shapeі
NameLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
NameLength/strided_slice/stackј
 NameLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 NameLength/strided_slice/stack_1ј
 NameLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 NameLength/strided_slice/stack_2ц
NameLength/strided_sliceStridedSliceNameLength/Shape:output:0'NameLength/strided_slice/stack:output:0)NameLength/strided_slice/stack_1:output:0)NameLength/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
NameLength/strided_slicez
NameLength/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
NameLength/Reshape/shape/1▓
NameLength/Reshape/shapePack!NameLength/strided_slice:output:0#NameLength/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
NameLength/Reshape/shapeћ
NameLength/ReshapeReshape
features_4!NameLength/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
NameLength/ReshapeV
Pclass/ShapeShape
features_5*
T0*
_output_shapes
:2
Pclass/Shapeѓ
Pclass/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Pclass/strided_slice/stackє
Pclass/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Pclass/strided_slice/stack_1є
Pclass/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Pclass/strided_slice/stack_2ї
Pclass/strided_sliceStridedSlicePclass/Shape:output:0#Pclass/strided_slice/stack:output:0%Pclass/strided_slice/stack_1:output:0%Pclass/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Pclass/strided_slicer
Pclass/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Pclass/Reshape/shape/1б
Pclass/Reshape/shapePackPclass/strided_slice:output:0Pclass/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Pclass/Reshape/shapeѕ
Pclass/ReshapeReshape
features_5Pclass/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Pclass/Reshapei
Sex/CastCast
features_6*

DstT0*

SrcT0	*'
_output_shapes
:         2

Sex/CastR
	Sex/ShapeShapeSex/Cast:y:0*
T0*
_output_shapes
:2
	Sex/Shape|
Sex/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Sex/strided_slice/stackђ
Sex/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Sex/strided_slice/stack_1ђ
Sex/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Sex/strided_slice/stack_2Щ
Sex/strided_sliceStridedSliceSex/Shape:output:0 Sex/strided_slice/stack:output:0"Sex/strided_slice/stack_1:output:0"Sex/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Sex/strided_slicel
Sex/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Sex/Reshape/shape/1ќ
Sex/Reshape/shapePackSex/strided_slice:output:0Sex/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Sex/Reshape/shapeЂ
Sex/ReshapeReshapeSex/Cast:y:0Sex/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Sex/ReshapeT
Title/ShapeShape
features_7*
T0*
_output_shapes
:2
Title/Shapeђ
Title/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Title/strided_slice/stackё
Title/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Title/strided_slice/stack_1ё
Title/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Title/strided_slice/stack_2є
Title/strided_sliceStridedSliceTitle/Shape:output:0"Title/strided_slice/stack:output:0$Title/strided_slice/stack_1:output:0$Title/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Title/strided_slicep
Title/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Title/Reshape/shape/1ъ
Title/Reshape/shapePackTitle/strided_slice:output:0Title/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Title/Reshape/shapeЁ
Title/ReshapeReshape
features_7Title/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Title/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2
concat/axis│
concatConcatV2Age/Reshape:output:0Embarked/Reshape:output:0Fare/Reshape:output:0IsAlone/Reshape:output:0NameLength/Reshape:output:0Pclass/Reshape:output:0Sex/Reshape:output:0Title/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesЏ
ў:         :         :         :         :         :         :         :         :Q M
'
_output_shapes
:         
"
_user_specified_name
features:QM
'
_output_shapes
:         
"
_user_specified_name
features:QM
'
_output_shapes
:         
"
_user_specified_name
features:QM
'
_output_shapes
:         
"
_user_specified_name
features:QM
'
_output_shapes
:         
"
_user_specified_name
features:QM
'
_output_shapes
:         
"
_user_specified_name
features:QM
'
_output_shapes
:         
"
_user_specified_name
features:QM
'
_output_shapes
:         
"
_user_specified_name
features
І
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_212770

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ▓2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ▓*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ▓2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ▓2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ▓2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ▓2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ▓:P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs
Є
d
+__inference_dropout_16_layer_call_fn_213614

inputs
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ъ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_2126462
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ъ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ъ22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs
І
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_213604

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ъ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ъ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ъ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ъ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ъ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ъ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ъ:P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs
хY
ѓ
L__inference_dense_features_4_layer_call_and_return_conditional_losses_213560
features_age
features_embarked
features_fare
features_isalone	
features_namelength
features_pclass
features_sex	
features_title
identityR
	Age/ShapeShapefeatures_age*
T0*
_output_shapes
:2
	Age/Shape|
Age/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Age/strided_slice/stackђ
Age/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Age/strided_slice/stack_1ђ
Age/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Age/strided_slice/stack_2Щ
Age/strided_sliceStridedSliceAge/Shape:output:0 Age/strided_slice/stack:output:0"Age/strided_slice/stack_1:output:0"Age/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Age/strided_slicel
Age/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Age/Reshape/shape/1ќ
Age/Reshape/shapePackAge/strided_slice:output:0Age/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Age/Reshape/shapeЂ
Age/ReshapeReshapefeatures_ageAge/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Age/Reshapea
Embarked/ShapeShapefeatures_embarked*
T0*
_output_shapes
:2
Embarked/Shapeє
Embarked/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Embarked/strided_slice/stackі
Embarked/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2 
Embarked/strided_slice/stack_1і
Embarked/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2 
Embarked/strided_slice/stack_2ў
Embarked/strided_sliceStridedSliceEmbarked/Shape:output:0%Embarked/strided_slice/stack:output:0'Embarked/strided_slice/stack_1:output:0'Embarked/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Embarked/strided_slicev
Embarked/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Embarked/Reshape/shape/1ф
Embarked/Reshape/shapePackEmbarked/strided_slice:output:0!Embarked/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Embarked/Reshape/shapeЋ
Embarked/ReshapeReshapefeatures_embarkedEmbarked/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Embarked/ReshapeU

Fare/ShapeShapefeatures_fare*
T0*
_output_shapes
:2

Fare/Shape~
Fare/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Fare/strided_slice/stackѓ
Fare/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Fare/strided_slice/stack_1ѓ
Fare/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Fare/strided_slice/stack_2ђ
Fare/strided_sliceStridedSliceFare/Shape:output:0!Fare/strided_slice/stack:output:0#Fare/strided_slice/stack_1:output:0#Fare/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Fare/strided_slicen
Fare/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Fare/Reshape/shape/1џ
Fare/Reshape/shapePackFare/strided_slice:output:0Fare/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Fare/Reshape/shapeЁ
Fare/ReshapeReshapefeatures_fareFare/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Fare/Reshapew
IsAlone/CastCastfeatures_isalone*

DstT0*

SrcT0	*'
_output_shapes
:         2
IsAlone/Cast^
IsAlone/ShapeShapeIsAlone/Cast:y:0*
T0*
_output_shapes
:2
IsAlone/Shapeё
IsAlone/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
IsAlone/strided_slice/stackѕ
IsAlone/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
IsAlone/strided_slice/stack_1ѕ
IsAlone/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
IsAlone/strided_slice/stack_2њ
IsAlone/strided_sliceStridedSliceIsAlone/Shape:output:0$IsAlone/strided_slice/stack:output:0&IsAlone/strided_slice/stack_1:output:0&IsAlone/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
IsAlone/strided_slicet
IsAlone/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
IsAlone/Reshape/shape/1д
IsAlone/Reshape/shapePackIsAlone/strided_slice:output:0 IsAlone/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
IsAlone/Reshape/shapeЉ
IsAlone/ReshapeReshapeIsAlone/Cast:y:0IsAlone/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
IsAlone/Reshapeg
NameLength/ShapeShapefeatures_namelength*
T0*
_output_shapes
:2
NameLength/Shapeі
NameLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2 
NameLength/strided_slice/stackј
 NameLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2"
 NameLength/strided_slice/stack_1ј
 NameLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2"
 NameLength/strided_slice/stack_2ц
NameLength/strided_sliceStridedSliceNameLength/Shape:output:0'NameLength/strided_slice/stack:output:0)NameLength/strided_slice/stack_1:output:0)NameLength/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
NameLength/strided_slicez
NameLength/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
NameLength/Reshape/shape/1▓
NameLength/Reshape/shapePack!NameLength/strided_slice:output:0#NameLength/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
NameLength/Reshape/shapeЮ
NameLength/ReshapeReshapefeatures_namelength!NameLength/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
NameLength/Reshape[
Pclass/ShapeShapefeatures_pclass*
T0*
_output_shapes
:2
Pclass/Shapeѓ
Pclass/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Pclass/strided_slice/stackє
Pclass/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Pclass/strided_slice/stack_1є
Pclass/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Pclass/strided_slice/stack_2ї
Pclass/strided_sliceStridedSlicePclass/Shape:output:0#Pclass/strided_slice/stack:output:0%Pclass/strided_slice/stack_1:output:0%Pclass/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Pclass/strided_slicer
Pclass/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Pclass/Reshape/shape/1б
Pclass/Reshape/shapePackPclass/strided_slice:output:0Pclass/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Pclass/Reshape/shapeЇ
Pclass/ReshapeReshapefeatures_pclassPclass/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Pclass/Reshapek
Sex/CastCastfeatures_sex*

DstT0*

SrcT0	*'
_output_shapes
:         2

Sex/CastR
	Sex/ShapeShapeSex/Cast:y:0*
T0*
_output_shapes
:2
	Sex/Shape|
Sex/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Sex/strided_slice/stackђ
Sex/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Sex/strided_slice/stack_1ђ
Sex/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Sex/strided_slice/stack_2Щ
Sex/strided_sliceStridedSliceSex/Shape:output:0 Sex/strided_slice/stack:output:0"Sex/strided_slice/stack_1:output:0"Sex/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Sex/strided_slicel
Sex/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Sex/Reshape/shape/1ќ
Sex/Reshape/shapePackSex/strided_slice:output:0Sex/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Sex/Reshape/shapeЂ
Sex/ReshapeReshapeSex/Cast:y:0Sex/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Sex/ReshapeX
Title/ShapeShapefeatures_title*
T0*
_output_shapes
:2
Title/Shapeђ
Title/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
Title/strided_slice/stackё
Title/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
Title/strided_slice/stack_1ё
Title/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
Title/strided_slice/stack_2є
Title/strided_sliceStridedSliceTitle/Shape:output:0"Title/strided_slice/stack:output:0$Title/strided_slice/stack_1:output:0$Title/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
Title/strided_slicep
Title/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Title/Reshape/shape/1ъ
Title/Reshape/shapePackTitle/strided_slice:output:0Title/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Title/Reshape/shapeЅ
Title/ReshapeReshapefeatures_titleTitle/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
Title/Reshapee
concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2
concat/axis│
concatConcatV2Age/Reshape:output:0Embarked/Reshape:output:0Fare/Reshape:output:0IsAlone/Reshape:output:0NameLength/Reshape:output:0Pclass/Reshape:output:0Sex/Reshape:output:0Title/Reshape:output:0concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesЏ
ў:         :         :         :         :         :         :         :         :U Q
'
_output_shapes
:         
&
_user_specified_namefeatures/Age:ZV
'
_output_shapes
:         
+
_user_specified_namefeatures/Embarked:VR
'
_output_shapes
:         
'
_user_specified_namefeatures/Fare:YU
'
_output_shapes
:         
*
_user_specified_namefeatures/IsAlone:\X
'
_output_shapes
:         
-
_user_specified_namefeatures/NameLength:XT
'
_output_shapes
:         
)
_user_specified_namefeatures/Pclass:UQ
'
_output_shapes
:         
&
_user_specified_namefeatures/Sex:WS
'
_output_shapes
:         
(
_user_specified_namefeatures/Title
Ь
г
D__inference_dense_23_layer_call_and_return_conditional_losses_213734

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
▓╝*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╝*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:         ╝2
Selug
IdentityIdentitySelu:activations:0*
T0*(
_output_shapes
:         ╝2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ▓:::P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ч
G
+__inference_dropout_16_layer_call_fn_213619

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ъ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_2126512
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ъ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ъ:P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs
ч
G
+__inference_dropout_19_layer_call_fn_213770

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ╝* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_2128322
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ╝2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╝:P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs
э>
я
H__inference_sequential_4_layer_call_and_return_conditional_losses_212977

inputs
inputs_1
inputs_2
inputs_3	
inputs_4
inputs_5
inputs_6	
inputs_7
dense_20_212947
dense_20_212949
dense_21_212953
dense_21_212955
dense_22_212959
dense_22_212961
dense_23_212965
dense_23_212967
dense_24_212971
dense_24_212973
identityѕб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCallб"dropout_16/StatefulPartitionedCallб"dropout_17/StatefulPartitionedCallб"dropout_18/StatefulPartitionedCallб"dropout_19/StatefulPartitionedCall
dense_features_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/CastЁ
dense_features_4/Cast_1Castinputs_1*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_1Ё
dense_features_4/Cast_2Castinputs_2*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_2Ё
dense_features_4/Cast_3Castinputs_4*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_3Ё
dense_features_4/Cast_4Castinputs_5*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_4Ё
dense_features_4/Cast_5Castinputs_7*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_5ї
 dense_features_4/PartitionedCallPartitionedCalldense_features_4/Cast:y:0dense_features_4/Cast_1:y:0dense_features_4/Cast_2:y:0inputs_3dense_features_4/Cast_3:y:0dense_features_4/Cast_4:y:0inputs_6dense_features_4/Cast_5:y:0*
Tin

2		*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_dense_features_4_layer_call_and_return_conditional_losses_2125922"
 dense_features_4/PartitionedCallЎ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_features_4/PartitionedCall:output:0dense_20_212947dense_20_212949*
Tin
2*
Tout
2*(
_output_shapes
:         ъ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_2126182"
 dense_20/StatefulPartitionedCallэ
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ъ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_2126462$
"dropout_16/StatefulPartitionedCallЏ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_21_212953dense_21_212955*
Tin
2*
Tout
2*(
_output_shapes
:         е*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_2126802"
 dense_21/StatefulPartitionedCallю
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         е* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_2127082$
"dropout_17/StatefulPartitionedCallЏ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_22_212959dense_22_212961*
Tin
2*
Tout
2*(
_output_shapes
:         ▓*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_2127422"
 dense_22/StatefulPartitionedCallю
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ▓* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_2127702$
"dropout_18/StatefulPartitionedCallЏ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_23_212965dense_23_212967*
Tin
2*
Tout
2*(
_output_shapes
:         ╝*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_2127992"
 dense_23/StatefulPartitionedCallю
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ╝* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_2128272$
"dropout_19/StatefulPartitionedCallџ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_24_212971dense_24_212973*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2128562"
 dense_24/StatefulPartitionedCall└
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
И8
╩
H__inference_sequential_4_layer_call_and_return_conditional_losses_213056

inputs
inputs_1
inputs_2
inputs_3	
inputs_4
inputs_5
inputs_6	
inputs_7
dense_20_213026
dense_20_213028
dense_21_213032
dense_21_213034
dense_22_213038
dense_22_213040
dense_23_213044
dense_23_213046
dense_24_213050
dense_24_213052
identityѕб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCall
dense_features_4/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/CastЁ
dense_features_4/Cast_1Castinputs_1*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_1Ё
dense_features_4/Cast_2Castinputs_2*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_2Ё
dense_features_4/Cast_3Castinputs_4*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_3Ё
dense_features_4/Cast_4Castinputs_5*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_4Ё
dense_features_4/Cast_5Castinputs_7*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_5ї
 dense_features_4/PartitionedCallPartitionedCalldense_features_4/Cast:y:0dense_features_4/Cast_1:y:0dense_features_4/Cast_2:y:0inputs_3dense_features_4/Cast_3:y:0dense_features_4/Cast_4:y:0inputs_6dense_features_4/Cast_5:y:0*
Tin

2		*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_dense_features_4_layer_call_and_return_conditional_losses_2125922"
 dense_features_4/PartitionedCallЎ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_features_4/PartitionedCall:output:0dense_20_213026dense_20_213028*
Tin
2*
Tout
2*(
_output_shapes
:         ъ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_2126182"
 dense_20/StatefulPartitionedCall▀
dropout_16/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ъ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_2126512
dropout_16/PartitionedCallЊ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_21_213032dense_21_213034*
Tin
2*
Tout
2*(
_output_shapes
:         е*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_2126802"
 dense_21/StatefulPartitionedCall▀
dropout_17/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         е* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_2127132
dropout_17/PartitionedCallЊ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_22_213038dense_22_213040*
Tin
2*
Tout
2*(
_output_shapes
:         ▓*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_2127422"
 dense_22/StatefulPartitionedCall▀
dropout_18/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ▓* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_2127752
dropout_18/PartitionedCallЊ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_23_213044dense_23_213046*
Tin
2*
Tout
2*(
_output_shapes
:         ╝*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_2127992"
 dense_23/StatefulPartitionedCall▀
dropout_19/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ╝* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_2128322
dropout_19/PartitionedCallњ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_24_213050dense_24_213052*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2128562"
 dense_24/StatefulPartitionedCallг
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:OK
'
_output_shapes
:         
 
_user_specified_nameinputs:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
І
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_212827

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╝2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╝*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╝2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╝2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╝2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╝2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╝:P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs
ч
~
)__inference_dense_24_layer_call_fn_213790

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallН
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2128562
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╝::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ь
г
D__inference_dense_23_layer_call_and_return_conditional_losses_212799

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
▓╝*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:╝*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:         ╝2
Selug
IdentityIdentitySelu:activations:0*
T0*(
_output_shapes
:         ╝2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ▓:::P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
§
~
)__inference_dense_23_layer_call_fn_213743

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         ╝*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_2127992
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╝2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ▓::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Є
d
+__inference_dropout_19_layer_call_fn_213765

inputs
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ╝* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_2128272
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ╝2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╝22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs
§
~
)__inference_dense_22_layer_call_fn_213696

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         ▓*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_2127422
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ▓2

Identity"
identityIdentity:output:0*/
_input_shapes
:         е::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         е
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
кл
█
!__inference__wrapped_model_212496
age
embarked
fare
isalone	

namelength

pclass
sex		
title8
4sequential_4_dense_20_matmul_readvariableop_resource9
5sequential_4_dense_20_biasadd_readvariableop_resource8
4sequential_4_dense_21_matmul_readvariableop_resource9
5sequential_4_dense_21_biasadd_readvariableop_resource8
4sequential_4_dense_22_matmul_readvariableop_resource9
5sequential_4_dense_22_biasadd_readvariableop_resource8
4sequential_4_dense_23_matmul_readvariableop_resource9
5sequential_4_dense_23_biasadd_readvariableop_resource8
4sequential_4_dense_24_matmul_readvariableop_resource9
5sequential_4_dense_24_biasadd_readvariableop_resource
identityѕќ
"sequential_4/dense_features_4/CastCastage*

DstT0*

SrcT0*'
_output_shapes
:         2$
"sequential_4/dense_features_4/CastЪ
$sequential_4/dense_features_4/Cast_1Castembarked*

DstT0*

SrcT0*'
_output_shapes
:         2&
$sequential_4/dense_features_4/Cast_1Џ
$sequential_4/dense_features_4/Cast_2Castfare*

DstT0*

SrcT0*'
_output_shapes
:         2&
$sequential_4/dense_features_4/Cast_2А
$sequential_4/dense_features_4/Cast_3Cast
namelength*

DstT0*

SrcT0*'
_output_shapes
:         2&
$sequential_4/dense_features_4/Cast_3Ю
$sequential_4/dense_features_4/Cast_4Castpclass*

DstT0*

SrcT0*'
_output_shapes
:         2&
$sequential_4/dense_features_4/Cast_4ю
$sequential_4/dense_features_4/Cast_5Casttitle*

DstT0*

SrcT0*'
_output_shapes
:         2&
$sequential_4/dense_features_4/Cast_5е
'sequential_4/dense_features_4/Age/ShapeShape&sequential_4/dense_features_4/Cast:y:0*
T0*
_output_shapes
:2)
'sequential_4/dense_features_4/Age/ShapeИ
5sequential_4/dense_features_4/Age/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_4/dense_features_4/Age/strided_slice/stack╝
7sequential_4/dense_features_4/Age/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_4/dense_features_4/Age/strided_slice/stack_1╝
7sequential_4/dense_features_4/Age/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_4/dense_features_4/Age/strided_slice/stack_2«
/sequential_4/dense_features_4/Age/strided_sliceStridedSlice0sequential_4/dense_features_4/Age/Shape:output:0>sequential_4/dense_features_4/Age/strided_slice/stack:output:0@sequential_4/dense_features_4/Age/strided_slice/stack_1:output:0@sequential_4/dense_features_4/Age/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_4/dense_features_4/Age/strided_sliceе
1sequential_4/dense_features_4/Age/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential_4/dense_features_4/Age/Reshape/shape/1ј
/sequential_4/dense_features_4/Age/Reshape/shapePack8sequential_4/dense_features_4/Age/strided_slice:output:0:sequential_4/dense_features_4/Age/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/sequential_4/dense_features_4/Age/Reshape/shapeш
)sequential_4/dense_features_4/Age/ReshapeReshape&sequential_4/dense_features_4/Cast:y:08sequential_4/dense_features_4/Age/Reshape/shape:output:0*
T0*'
_output_shapes
:         2+
)sequential_4/dense_features_4/Age/Reshape┤
,sequential_4/dense_features_4/Embarked/ShapeShape(sequential_4/dense_features_4/Cast_1:y:0*
T0*
_output_shapes
:2.
,sequential_4/dense_features_4/Embarked/Shape┬
:sequential_4/dense_features_4/Embarked/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:sequential_4/dense_features_4/Embarked/strided_slice/stackк
<sequential_4/dense_features_4/Embarked/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_4/dense_features_4/Embarked/strided_slice/stack_1к
<sequential_4/dense_features_4/Embarked/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_4/dense_features_4/Embarked/strided_slice/stack_2╠
4sequential_4/dense_features_4/Embarked/strided_sliceStridedSlice5sequential_4/dense_features_4/Embarked/Shape:output:0Csequential_4/dense_features_4/Embarked/strided_slice/stack:output:0Esequential_4/dense_features_4/Embarked/strided_slice/stack_1:output:0Esequential_4/dense_features_4/Embarked/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4sequential_4/dense_features_4/Embarked/strided_slice▓
6sequential_4/dense_features_4/Embarked/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :28
6sequential_4/dense_features_4/Embarked/Reshape/shape/1б
4sequential_4/dense_features_4/Embarked/Reshape/shapePack=sequential_4/dense_features_4/Embarked/strided_slice:output:0?sequential_4/dense_features_4/Embarked/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:26
4sequential_4/dense_features_4/Embarked/Reshape/shapeє
.sequential_4/dense_features_4/Embarked/ReshapeReshape(sequential_4/dense_features_4/Cast_1:y:0=sequential_4/dense_features_4/Embarked/Reshape/shape:output:0*
T0*'
_output_shapes
:         20
.sequential_4/dense_features_4/Embarked/Reshapeг
(sequential_4/dense_features_4/Fare/ShapeShape(sequential_4/dense_features_4/Cast_2:y:0*
T0*
_output_shapes
:2*
(sequential_4/dense_features_4/Fare/Shape║
6sequential_4/dense_features_4/Fare/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6sequential_4/dense_features_4/Fare/strided_slice/stackЙ
8sequential_4/dense_features_4/Fare/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_4/dense_features_4/Fare/strided_slice/stack_1Й
8sequential_4/dense_features_4/Fare/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8sequential_4/dense_features_4/Fare/strided_slice/stack_2┤
0sequential_4/dense_features_4/Fare/strided_sliceStridedSlice1sequential_4/dense_features_4/Fare/Shape:output:0?sequential_4/dense_features_4/Fare/strided_slice/stack:output:0Asequential_4/dense_features_4/Fare/strided_slice/stack_1:output:0Asequential_4/dense_features_4/Fare/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask22
0sequential_4/dense_features_4/Fare/strided_sliceф
2sequential_4/dense_features_4/Fare/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :24
2sequential_4/dense_features_4/Fare/Reshape/shape/1њ
0sequential_4/dense_features_4/Fare/Reshape/shapePack9sequential_4/dense_features_4/Fare/strided_slice:output:0;sequential_4/dense_features_4/Fare/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:22
0sequential_4/dense_features_4/Fare/Reshape/shapeЩ
*sequential_4/dense_features_4/Fare/ReshapeReshape(sequential_4/dense_features_4/Cast_2:y:09sequential_4/dense_features_4/Fare/Reshape/shape:output:0*
T0*'
_output_shapes
:         2,
*sequential_4/dense_features_4/Fare/Reshapeф
*sequential_4/dense_features_4/IsAlone/CastCastisalone*

DstT0*

SrcT0	*'
_output_shapes
:         2,
*sequential_4/dense_features_4/IsAlone/CastИ
+sequential_4/dense_features_4/IsAlone/ShapeShape.sequential_4/dense_features_4/IsAlone/Cast:y:0*
T0*
_output_shapes
:2-
+sequential_4/dense_features_4/IsAlone/Shape└
9sequential_4/dense_features_4/IsAlone/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2;
9sequential_4/dense_features_4/IsAlone/strided_slice/stack─
;sequential_4/dense_features_4/IsAlone/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential_4/dense_features_4/IsAlone/strided_slice/stack_1─
;sequential_4/dense_features_4/IsAlone/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2=
;sequential_4/dense_features_4/IsAlone/strided_slice/stack_2к
3sequential_4/dense_features_4/IsAlone/strided_sliceStridedSlice4sequential_4/dense_features_4/IsAlone/Shape:output:0Bsequential_4/dense_features_4/IsAlone/strided_slice/stack:output:0Dsequential_4/dense_features_4/IsAlone/strided_slice/stack_1:output:0Dsequential_4/dense_features_4/IsAlone/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask25
3sequential_4/dense_features_4/IsAlone/strided_slice░
5sequential_4/dense_features_4/IsAlone/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :27
5sequential_4/dense_features_4/IsAlone/Reshape/shape/1ъ
3sequential_4/dense_features_4/IsAlone/Reshape/shapePack<sequential_4/dense_features_4/IsAlone/strided_slice:output:0>sequential_4/dense_features_4/IsAlone/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:25
3sequential_4/dense_features_4/IsAlone/Reshape/shapeЅ
-sequential_4/dense_features_4/IsAlone/ReshapeReshape.sequential_4/dense_features_4/IsAlone/Cast:y:0<sequential_4/dense_features_4/IsAlone/Reshape/shape:output:0*
T0*'
_output_shapes
:         2/
-sequential_4/dense_features_4/IsAlone/ReshapeИ
.sequential_4/dense_features_4/NameLength/ShapeShape(sequential_4/dense_features_4/Cast_3:y:0*
T0*
_output_shapes
:20
.sequential_4/dense_features_4/NameLength/Shapeк
<sequential_4/dense_features_4/NameLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<sequential_4/dense_features_4/NameLength/strided_slice/stack╩
>sequential_4/dense_features_4/NameLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_4/dense_features_4/NameLength/strided_slice/stack_1╩
>sequential_4/dense_features_4/NameLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_4/dense_features_4/NameLength/strided_slice/stack_2п
6sequential_4/dense_features_4/NameLength/strided_sliceStridedSlice7sequential_4/dense_features_4/NameLength/Shape:output:0Esequential_4/dense_features_4/NameLength/strided_slice/stack:output:0Gsequential_4/dense_features_4/NameLength/strided_slice/stack_1:output:0Gsequential_4/dense_features_4/NameLength/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6sequential_4/dense_features_4/NameLength/strided_sliceХ
8sequential_4/dense_features_4/NameLength/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2:
8sequential_4/dense_features_4/NameLength/Reshape/shape/1ф
6sequential_4/dense_features_4/NameLength/Reshape/shapePack?sequential_4/dense_features_4/NameLength/strided_slice:output:0Asequential_4/dense_features_4/NameLength/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:28
6sequential_4/dense_features_4/NameLength/Reshape/shapeї
0sequential_4/dense_features_4/NameLength/ReshapeReshape(sequential_4/dense_features_4/Cast_3:y:0?sequential_4/dense_features_4/NameLength/Reshape/shape:output:0*
T0*'
_output_shapes
:         22
0sequential_4/dense_features_4/NameLength/Reshape░
*sequential_4/dense_features_4/Pclass/ShapeShape(sequential_4/dense_features_4/Cast_4:y:0*
T0*
_output_shapes
:2,
*sequential_4/dense_features_4/Pclass/ShapeЙ
8sequential_4/dense_features_4/Pclass/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8sequential_4/dense_features_4/Pclass/strided_slice/stack┬
:sequential_4/dense_features_4/Pclass/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_4/dense_features_4/Pclass/strided_slice/stack_1┬
:sequential_4/dense_features_4/Pclass/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:sequential_4/dense_features_4/Pclass/strided_slice/stack_2└
2sequential_4/dense_features_4/Pclass/strided_sliceStridedSlice3sequential_4/dense_features_4/Pclass/Shape:output:0Asequential_4/dense_features_4/Pclass/strided_slice/stack:output:0Csequential_4/dense_features_4/Pclass/strided_slice/stack_1:output:0Csequential_4/dense_features_4/Pclass/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2sequential_4/dense_features_4/Pclass/strided_slice«
4sequential_4/dense_features_4/Pclass/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4sequential_4/dense_features_4/Pclass/Reshape/shape/1џ
2sequential_4/dense_features_4/Pclass/Reshape/shapePack;sequential_4/dense_features_4/Pclass/strided_slice:output:0=sequential_4/dense_features_4/Pclass/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:24
2sequential_4/dense_features_4/Pclass/Reshape/shapeђ
,sequential_4/dense_features_4/Pclass/ReshapeReshape(sequential_4/dense_features_4/Cast_4:y:0;sequential_4/dense_features_4/Pclass/Reshape/shape:output:0*
T0*'
_output_shapes
:         2.
,sequential_4/dense_features_4/Pclass/Reshapeъ
&sequential_4/dense_features_4/Sex/CastCastsex*

DstT0*

SrcT0	*'
_output_shapes
:         2(
&sequential_4/dense_features_4/Sex/Castг
'sequential_4/dense_features_4/Sex/ShapeShape*sequential_4/dense_features_4/Sex/Cast:y:0*
T0*
_output_shapes
:2)
'sequential_4/dense_features_4/Sex/ShapeИ
5sequential_4/dense_features_4/Sex/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_4/dense_features_4/Sex/strided_slice/stack╝
7sequential_4/dense_features_4/Sex/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_4/dense_features_4/Sex/strided_slice/stack_1╝
7sequential_4/dense_features_4/Sex/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7sequential_4/dense_features_4/Sex/strided_slice/stack_2«
/sequential_4/dense_features_4/Sex/strided_sliceStridedSlice0sequential_4/dense_features_4/Sex/Shape:output:0>sequential_4/dense_features_4/Sex/strided_slice/stack:output:0@sequential_4/dense_features_4/Sex/strided_slice/stack_1:output:0@sequential_4/dense_features_4/Sex/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/sequential_4/dense_features_4/Sex/strided_sliceе
1sequential_4/dense_features_4/Sex/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1sequential_4/dense_features_4/Sex/Reshape/shape/1ј
/sequential_4/dense_features_4/Sex/Reshape/shapePack8sequential_4/dense_features_4/Sex/strided_slice:output:0:sequential_4/dense_features_4/Sex/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:21
/sequential_4/dense_features_4/Sex/Reshape/shapeщ
)sequential_4/dense_features_4/Sex/ReshapeReshape*sequential_4/dense_features_4/Sex/Cast:y:08sequential_4/dense_features_4/Sex/Reshape/shape:output:0*
T0*'
_output_shapes
:         2+
)sequential_4/dense_features_4/Sex/Reshape«
)sequential_4/dense_features_4/Title/ShapeShape(sequential_4/dense_features_4/Cast_5:y:0*
T0*
_output_shapes
:2+
)sequential_4/dense_features_4/Title/Shape╝
7sequential_4/dense_features_4/Title/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 29
7sequential_4/dense_features_4/Title/strided_slice/stack└
9sequential_4/dense_features_4/Title/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_4/dense_features_4/Title/strided_slice/stack_1└
9sequential_4/dense_features_4/Title/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2;
9sequential_4/dense_features_4/Title/strided_slice/stack_2║
1sequential_4/dense_features_4/Title/strided_sliceStridedSlice2sequential_4/dense_features_4/Title/Shape:output:0@sequential_4/dense_features_4/Title/strided_slice/stack:output:0Bsequential_4/dense_features_4/Title/strided_slice/stack_1:output:0Bsequential_4/dense_features_4/Title/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask23
1sequential_4/dense_features_4/Title/strided_sliceг
3sequential_4/dense_features_4/Title/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3sequential_4/dense_features_4/Title/Reshape/shape/1ќ
1sequential_4/dense_features_4/Title/Reshape/shapePack:sequential_4/dense_features_4/Title/strided_slice:output:0<sequential_4/dense_features_4/Title/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:23
1sequential_4/dense_features_4/Title/Reshape/shape§
+sequential_4/dense_features_4/Title/ReshapeReshape(sequential_4/dense_features_4/Cast_5:y:0:sequential_4/dense_features_4/Title/Reshape/shape:output:0*
T0*'
_output_shapes
:         2-
+sequential_4/dense_features_4/Title/ReshapeА
)sequential_4/dense_features_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2+
)sequential_4/dense_features_4/concat/axis§
$sequential_4/dense_features_4/concatConcatV22sequential_4/dense_features_4/Age/Reshape:output:07sequential_4/dense_features_4/Embarked/Reshape:output:03sequential_4/dense_features_4/Fare/Reshape:output:06sequential_4/dense_features_4/IsAlone/Reshape:output:09sequential_4/dense_features_4/NameLength/Reshape:output:05sequential_4/dense_features_4/Pclass/Reshape:output:02sequential_4/dense_features_4/Sex/Reshape:output:04sequential_4/dense_features_4/Title/Reshape:output:02sequential_4/dense_features_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2&
$sequential_4/dense_features_4/concatл
+sequential_4/dense_20/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_20_matmul_readvariableop_resource*
_output_shapes
:	ъ*
dtype02-
+sequential_4/dense_20/MatMul/ReadVariableOpП
sequential_4/dense_20/MatMulMatMul-sequential_4/dense_features_4/concat:output:03sequential_4/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
sequential_4/dense_20/MatMul¤
,sequential_4/dense_20/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:ъ*
dtype02.
,sequential_4/dense_20/BiasAdd/ReadVariableOp┌
sequential_4/dense_20/BiasAddBiasAdd&sequential_4/dense_20/MatMul:product:04sequential_4/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
sequential_4/dense_20/BiasAddЏ
sequential_4/dense_20/SeluSelu&sequential_4/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:         ъ2
sequential_4/dense_20/SeluГ
 sequential_4/dropout_16/IdentityIdentity(sequential_4/dense_20/Selu:activations:0*
T0*(
_output_shapes
:         ъ2"
 sequential_4/dropout_16/IdentityЛ
+sequential_4/dense_21/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ъе*
dtype02-
+sequential_4/dense_21/MatMul/ReadVariableOp┘
sequential_4/dense_21/MatMulMatMul)sequential_4/dropout_16/Identity:output:03sequential_4/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
sequential_4/dense_21/MatMul¤
,sequential_4/dense_21/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:е*
dtype02.
,sequential_4/dense_21/BiasAdd/ReadVariableOp┌
sequential_4/dense_21/BiasAddBiasAdd&sequential_4/dense_21/MatMul:product:04sequential_4/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
sequential_4/dense_21/BiasAddц
sequential_4/dense_21/SigmoidSigmoid&sequential_4/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         е2
sequential_4/dense_21/Sigmoid╗
sequential_4/dense_21/mulMul&sequential_4/dense_21/BiasAdd:output:0!sequential_4/dense_21/Sigmoid:y:0*
T0*(
_output_shapes
:         е2
sequential_4/dense_21/mulъ
sequential_4/dense_21/IdentityIdentitysequential_4/dense_21/mul:z:0*
T0*(
_output_shapes
:         е2 
sequential_4/dense_21/IdentityЈ
sequential_4/dense_21/IdentityN	IdentityNsequential_4/dense_21/mul:z:0&sequential_4/dense_21/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-212460*<
_output_shapes*
(:         е:         е2!
sequential_4/dense_21/IdentityNГ
 sequential_4/dropout_17/IdentityIdentity(sequential_4/dense_21/IdentityN:output:0*
T0*(
_output_shapes
:         е2"
 sequential_4/dropout_17/IdentityЛ
+sequential_4/dense_22/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
е▓*
dtype02-
+sequential_4/dense_22/MatMul/ReadVariableOp┘
sequential_4/dense_22/MatMulMatMul)sequential_4/dropout_17/Identity:output:03sequential_4/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
sequential_4/dense_22/MatMul¤
,sequential_4/dense_22/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:▓*
dtype02.
,sequential_4/dense_22/BiasAdd/ReadVariableOp┌
sequential_4/dense_22/BiasAddBiasAdd&sequential_4/dense_22/MatMul:product:04sequential_4/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
sequential_4/dense_22/BiasAddц
sequential_4/dense_22/SigmoidSigmoid&sequential_4/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ▓2
sequential_4/dense_22/Sigmoid╗
sequential_4/dense_22/mulMul&sequential_4/dense_22/BiasAdd:output:0!sequential_4/dense_22/Sigmoid:y:0*
T0*(
_output_shapes
:         ▓2
sequential_4/dense_22/mulъ
sequential_4/dense_22/IdentityIdentitysequential_4/dense_22/mul:z:0*
T0*(
_output_shapes
:         ▓2 
sequential_4/dense_22/IdentityЈ
sequential_4/dense_22/IdentityN	IdentityNsequential_4/dense_22/mul:z:0&sequential_4/dense_22/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-212473*<
_output_shapes*
(:         ▓:         ▓2!
sequential_4/dense_22/IdentityNГ
 sequential_4/dropout_18/IdentityIdentity(sequential_4/dense_22/IdentityN:output:0*
T0*(
_output_shapes
:         ▓2"
 sequential_4/dropout_18/IdentityЛ
+sequential_4/dense_23/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_23_matmul_readvariableop_resource* 
_output_shapes
:
▓╝*
dtype02-
+sequential_4/dense_23/MatMul/ReadVariableOp┘
sequential_4/dense_23/MatMulMatMul)sequential_4/dropout_18/Identity:output:03sequential_4/dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
sequential_4/dense_23/MatMul¤
,sequential_4/dense_23/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_23_biasadd_readvariableop_resource*
_output_shapes	
:╝*
dtype02.
,sequential_4/dense_23/BiasAdd/ReadVariableOp┌
sequential_4/dense_23/BiasAddBiasAdd&sequential_4/dense_23/MatMul:product:04sequential_4/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
sequential_4/dense_23/BiasAddЏ
sequential_4/dense_23/SeluSelu&sequential_4/dense_23/BiasAdd:output:0*
T0*(
_output_shapes
:         ╝2
sequential_4/dense_23/SeluГ
 sequential_4/dropout_19/IdentityIdentity(sequential_4/dense_23/Selu:activations:0*
T0*(
_output_shapes
:         ╝2"
 sequential_4/dropout_19/Identityл
+sequential_4/dense_24/MatMul/ReadVariableOpReadVariableOp4sequential_4_dense_24_matmul_readvariableop_resource*
_output_shapes
:	╝*
dtype02-
+sequential_4/dense_24/MatMul/ReadVariableOpп
sequential_4/dense_24/MatMulMatMul)sequential_4/dropout_19/Identity:output:03sequential_4/dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_4/dense_24/MatMul╬
,sequential_4/dense_24/BiasAdd/ReadVariableOpReadVariableOp5sequential_4_dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,sequential_4/dense_24/BiasAdd/ReadVariableOp┘
sequential_4/dense_24/BiasAddBiasAdd&sequential_4/dense_24/MatMul:product:04sequential_4/dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
sequential_4/dense_24/BiasAddБ
sequential_4/dense_24/SigmoidSigmoid&sequential_4/dense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
sequential_4/dense_24/Sigmoidu
IdentityIdentity!sequential_4/dense_24/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         :::::::::::L H
'
_output_shapes
:         

_user_specified_nameAge:QM
'
_output_shapes
:         
"
_user_specified_name
Embarked:MI
'
_output_shapes
:         

_user_specified_nameFare:PL
'
_output_shapes
:         
!
_user_specified_name	IsAlone:SO
'
_output_shapes
:         
$
_user_specified_name
NameLength:OK
'
_output_shapes
:         
 
_user_specified_namePclass:LH
'
_output_shapes
:         

_user_specified_nameSex:NJ
'
_output_shapes
:         

_user_specified_nameTitle:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ј
├
$__inference_signature_wrapper_213121
age
embarked
fare
isalone	

namelength

pclass
sex		
title
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityѕбStatefulPartitionedCallО
StatefulPartitionedCallStatefulPartitionedCallageembarkedfareisalone
namelengthpclasssextitleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__wrapped_model_2124962
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:         

_user_specified_nameAge:QM
'
_output_shapes
:         
"
_user_specified_name
Embarked:MI
'
_output_shapes
:         

_user_specified_nameFare:PL
'
_output_shapes
:         
!
_user_specified_name	IsAlone:SO
'
_output_shapes
:         
$
_user_specified_name
NameLength:OK
'
_output_shapes
:         
 
_user_specified_namePclass:LH
'
_output_shapes
:         

_user_specified_nameSex:NJ
'
_output_shapes
:         

_user_specified_nameTitle:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
═
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_212832

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╝2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╝2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╝:P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs
в
г
D__inference_dense_20_layer_call_and_return_conditional_losses_213583

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ъ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ъ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:         ъ2
Selug
IdentityIdentitySelu:activations:0*
T0*(
_output_shapes
:         ъ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Й
╠
-__inference_sequential_4_layer_call_fn_213000
age
embarked
fare
isalone	

namelength

pclass
sex		
title
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallageembarkedfareisalone
namelengthpclasssextitleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_2129772
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:         

_user_specified_nameAge:QM
'
_output_shapes
:         
"
_user_specified_name
Embarked:MI
'
_output_shapes
:         

_user_specified_nameFare:PL
'
_output_shapes
:         
!
_user_specified_name	IsAlone:SO
'
_output_shapes
:         
$
_user_specified_name
NameLength:OK
'
_output_shapes
:         
 
_user_specified_namePclass:LH
'
_output_shapes
:         

_user_specified_nameSex:NJ
'
_output_shapes
:         

_user_specified_nameTitle:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
«
D__inference_dense_22_layer_call_and_return_conditional_losses_212742

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
е▓*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:▓*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         ▓2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ▓2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:         ▓2

Identityи
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-212735*<
_output_shapes*
(:         ▓:         ▓2
	IdentityNk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:         ▓2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         е:::P L
(
_output_shapes
:         е
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
═
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_213609

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ъ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ъ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ъ:P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs
І
e
F__inference_dropout_16_layer_call_and_return_conditional_losses_212646

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ъ2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ъ*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ъ2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ъ2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ъ2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ъ2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ъ:P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs
Н>
╬
H__inference_sequential_4_layer_call_and_return_conditional_losses_212873
age
embarked
fare
isalone	

namelength

pclass
sex		
title
dense_20_212629
dense_20_212631
dense_21_212691
dense_21_212693
dense_22_212753
dense_22_212755
dense_23_212810
dense_23_212812
dense_24_212867
dense_24_212869
identityѕб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCallб"dropout_16/StatefulPartitionedCallб"dropout_17/StatefulPartitionedCallб"dropout_18/StatefulPartitionedCallб"dropout_19/StatefulPartitionedCall|
dense_features_4/CastCastage*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/CastЁ
dense_features_4/Cast_1Castembarked*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_1Ђ
dense_features_4/Cast_2Castfare*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_2Є
dense_features_4/Cast_3Cast
namelength*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_3Ѓ
dense_features_4/Cast_4Castpclass*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_4ѓ
dense_features_4/Cast_5Casttitle*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_5є
 dense_features_4/PartitionedCallPartitionedCalldense_features_4/Cast:y:0dense_features_4/Cast_1:y:0dense_features_4/Cast_2:y:0isalonedense_features_4/Cast_3:y:0dense_features_4/Cast_4:y:0sexdense_features_4/Cast_5:y:0*
Tin

2		*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_dense_features_4_layer_call_and_return_conditional_losses_2125922"
 dense_features_4/PartitionedCallЎ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_features_4/PartitionedCall:output:0dense_20_212629dense_20_212631*
Tin
2*
Tout
2*(
_output_shapes
:         ъ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_2126182"
 dense_20/StatefulPartitionedCallэ
"dropout_16/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ъ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_2126462$
"dropout_16/StatefulPartitionedCallЏ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall+dropout_16/StatefulPartitionedCall:output:0dense_21_212691dense_21_212693*
Tin
2*
Tout
2*(
_output_shapes
:         е*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_2126802"
 dense_21/StatefulPartitionedCallю
"dropout_17/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0#^dropout_16/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         е* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_2127082$
"dropout_17/StatefulPartitionedCallЏ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall+dropout_17/StatefulPartitionedCall:output:0dense_22_212753dense_22_212755*
Tin
2*
Tout
2*(
_output_shapes
:         ▓*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_2127422"
 dense_22/StatefulPartitionedCallю
"dropout_18/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0#^dropout_17/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ▓* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_2127702$
"dropout_18/StatefulPartitionedCallЏ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall+dropout_18/StatefulPartitionedCall:output:0dense_23_212810dense_23_212812*
Tin
2*
Tout
2*(
_output_shapes
:         ╝*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_2127992"
 dense_23/StatefulPartitionedCallю
"dropout_19/StatefulPartitionedCallStatefulPartitionedCall)dense_23/StatefulPartitionedCall:output:0#^dropout_18/StatefulPartitionedCall*
Tin
2*
Tout
2*(
_output_shapes
:         ╝* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_2128272$
"dropout_19/StatefulPartitionedCallџ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall+dropout_19/StatefulPartitionedCall:output:0dense_24_212867dense_24_212869*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2128562"
 dense_24/StatefulPartitionedCall└
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall#^dropout_16/StatefulPartitionedCall#^dropout_17/StatefulPartitionedCall#^dropout_18/StatefulPartitionedCall#^dropout_19/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall2H
"dropout_16/StatefulPartitionedCall"dropout_16/StatefulPartitionedCall2H
"dropout_17/StatefulPartitionedCall"dropout_17/StatefulPartitionedCall2H
"dropout_18/StatefulPartitionedCall"dropout_18/StatefulPartitionedCall2H
"dropout_19/StatefulPartitionedCall"dropout_19/StatefulPartitionedCall:L H
'
_output_shapes
:         

_user_specified_nameAge:QM
'
_output_shapes
:         
"
_user_specified_name
Embarked:MI
'
_output_shapes
:         

_user_specified_nameFare:PL
'
_output_shapes
:         
!
_user_specified_name	IsAlone:SO
'
_output_shapes
:         
$
_user_specified_name
NameLength:OK
'
_output_shapes
:         
 
_user_specified_namePclass:LH
'
_output_shapes
:         

_user_specified_nameSex:NJ
'
_output_shapes
:         

_user_specified_nameTitle:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Є
d
+__inference_dropout_17_layer_call_fn_213666

inputs
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         е* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_2127082
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         е2

Identity"
identityIdentity:output:0*'
_input_shapes
:         е22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         е
 
_user_specified_nameinputs
═
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_212713

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         е2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         е2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         е:P L
(
_output_shapes
:         е
 
_user_specified_nameinputs
І
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_213656

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         е2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         е*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         е2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         е2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         е2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         е2

Identity"
identityIdentity:output:0*'
_input_shapes
:         е:P L
(
_output_shapes
:         е
 
_user_specified_nameinputs
Ж
г
D__inference_dense_24_layer_call_and_return_conditional_losses_213781

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╝*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╝:::P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
в
г
D__inference_dense_20_layer_call_and_return_conditional_losses_212618

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	ъ*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:ъ*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2	
BiasAddY
SeluSeluBiasAdd:output:0*
T0*(
_output_shapes
:         ъ2
Selug
IdentityIdentitySelu:activations:0*
T0*(
_output_shapes
:         ъ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         :::O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Ж
г
D__inference_dense_24_layer_call_and_return_conditional_losses_212856

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕј
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	╝*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulї
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЂ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:         2	
Sigmoid_
IdentityIdentitySigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ╝:::P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
І
e
F__inference_dropout_17_layer_call_and_return_conditional_losses_212708

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         е2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         е*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         е2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         е2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         е2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         е2

Identity"
identityIdentity:output:0*'
_input_shapes
:         е:P L
(
_output_shapes
:         е
 
_user_specified_nameinputs
ч
~
)__inference_dense_20_layer_call_fn_213592

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         ъ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_2126182
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ъ2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
■
«
D__inference_dense_21_layer_call_and_return_conditional_losses_213635

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ъе*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:е*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         е2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         е2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:         е2

Identityи
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-213628*<
_output_shapes*
(:         е:         е2
	IdentityNk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:         е2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         ъ:::P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
Т
ё
-__inference_sequential_4_layer_call_fn_213449

inputs_age
inputs_embarked
inputs_fare
inputs_isalone	
inputs_namelength
inputs_pclass

inputs_sex	
inputs_title
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_embarkedinputs_fareinputs_isaloneinputs_namelengthinputs_pclass
inputs_sexinputs_titleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_2129772
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:         
$
_user_specified_name
inputs/Age:XT
'
_output_shapes
:         
)
_user_specified_nameinputs/Embarked:TP
'
_output_shapes
:         
%
_user_specified_nameinputs/Fare:WS
'
_output_shapes
:         
(
_user_specified_nameinputs/IsAlone:ZV
'
_output_shapes
:         
+
_user_specified_nameinputs/NameLength:VR
'
_output_shapes
:         
'
_user_specified_nameinputs/Pclass:SO
'
_output_shapes
:         
$
_user_specified_name
inputs/Sex:UQ
'
_output_shapes
:         
&
_user_specified_nameinputs/Title:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
е
у
1__inference_dense_features_4_layer_call_fn_213572
features_age
features_embarked
features_fare
features_isalone	
features_namelength
features_pclass
features_sex	
features_title
identity░
PartitionedCallPartitionedCallfeatures_agefeatures_embarkedfeatures_farefeatures_isalonefeatures_namelengthfeatures_pclassfeatures_sexfeatures_title*
Tin

2		*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_dense_features_4_layer_call_and_return_conditional_losses_2125922
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Г
_input_shapesЏ
ў:         :         :         :         :         :         :         :         :U Q
'
_output_shapes
:         
&
_user_specified_namefeatures/Age:ZV
'
_output_shapes
:         
+
_user_specified_namefeatures/Embarked:VR
'
_output_shapes
:         
'
_user_specified_namefeatures/Fare:YU
'
_output_shapes
:         
*
_user_specified_namefeatures/IsAlone:\X
'
_output_shapes
:         
-
_user_specified_namefeatures/NameLength:XT
'
_output_shapes
:         
)
_user_specified_namefeatures/Pclass:UQ
'
_output_shapes
:         
&
_user_specified_namefeatures/Sex:WS
'
_output_shapes
:         
(
_user_specified_namefeatures/Title
§
~
)__inference_dense_21_layer_call_fn_213644

inputs
unknown
	unknown_0
identityѕбStatefulPartitionedCallо
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*(
_output_shapes
:         е*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_2126802
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         е2

Identity"
identityIdentity:output:0*/
_input_shapes
:         ъ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
ч
G
+__inference_dropout_17_layer_call_fn_213671

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         е* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_2127132
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         е2

Identity"
identityIdentity:output:0*'
_input_shapes
:         е:P L
(
_output_shapes
:         е
 
_user_specified_nameinputs
═
d
F__inference_dropout_19_layer_call_and_return_conditional_losses_213760

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ╝2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ╝2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ╝:P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs
═
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_212775

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ▓2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ▓2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ▓:P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs
═
d
F__inference_dropout_17_layer_call_and_return_conditional_losses_213661

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         е2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         е2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         е:P L
(
_output_shapes
:         е
 
_user_specified_nameinputs
▀М
И
H__inference_sequential_4_layer_call_and_return_conditional_losses_213283

inputs_age
inputs_embarked
inputs_fare
inputs_isalone	
inputs_namelength
inputs_pclass

inputs_sex	
inputs_title+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identityѕЃ
dense_features_4/CastCast
inputs_age*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Castї
dense_features_4/Cast_1Castinputs_embarked*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_1ѕ
dense_features_4/Cast_2Castinputs_fare*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_2ј
dense_features_4/Cast_3Castinputs_namelength*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_3і
dense_features_4/Cast_4Castinputs_pclass*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_4Ѕ
dense_features_4/Cast_5Castinputs_title*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_5Ђ
dense_features_4/Age/ShapeShapedense_features_4/Cast:y:0*
T0*
_output_shapes
:2
dense_features_4/Age/Shapeъ
(dense_features_4/Age/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features_4/Age/strided_slice/stackб
*dense_features_4/Age/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Age/strided_slice/stack_1б
*dense_features_4/Age/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Age/strided_slice/stack_2Я
"dense_features_4/Age/strided_sliceStridedSlice#dense_features_4/Age/Shape:output:01dense_features_4/Age/strided_slice/stack:output:03dense_features_4/Age/strided_slice/stack_1:output:03dense_features_4/Age/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features_4/Age/strided_sliceј
$dense_features_4/Age/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features_4/Age/Reshape/shape/1┌
"dense_features_4/Age/Reshape/shapePack+dense_features_4/Age/strided_slice:output:0-dense_features_4/Age/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features_4/Age/Reshape/shape┴
dense_features_4/Age/ReshapeReshapedense_features_4/Cast:y:0+dense_features_4/Age/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
dense_features_4/Age/ReshapeЇ
dense_features_4/Embarked/ShapeShapedense_features_4/Cast_1:y:0*
T0*
_output_shapes
:2!
dense_features_4/Embarked/Shapeе
-dense_features_4/Embarked/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense_features_4/Embarked/strided_slice/stackг
/dense_features_4/Embarked/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features_4/Embarked/strided_slice/stack_1г
/dense_features_4/Embarked/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features_4/Embarked/strided_slice/stack_2■
'dense_features_4/Embarked/strided_sliceStridedSlice(dense_features_4/Embarked/Shape:output:06dense_features_4/Embarked/strided_slice/stack:output:08dense_features_4/Embarked/strided_slice/stack_1:output:08dense_features_4/Embarked/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense_features_4/Embarked/strided_sliceў
)dense_features_4/Embarked/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)dense_features_4/Embarked/Reshape/shape/1Ь
'dense_features_4/Embarked/Reshape/shapePack0dense_features_4/Embarked/strided_slice:output:02dense_features_4/Embarked/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2)
'dense_features_4/Embarked/Reshape/shapeм
!dense_features_4/Embarked/ReshapeReshapedense_features_4/Cast_1:y:00dense_features_4/Embarked/Reshape/shape:output:0*
T0*'
_output_shapes
:         2#
!dense_features_4/Embarked/ReshapeЁ
dense_features_4/Fare/ShapeShapedense_features_4/Cast_2:y:0*
T0*
_output_shapes
:2
dense_features_4/Fare/Shapeа
)dense_features_4/Fare/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features_4/Fare/strided_slice/stackц
+dense_features_4/Fare/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_4/Fare/strided_slice/stack_1ц
+dense_features_4/Fare/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_4/Fare/strided_slice/stack_2Т
#dense_features_4/Fare/strided_sliceStridedSlice$dense_features_4/Fare/Shape:output:02dense_features_4/Fare/strided_slice/stack:output:04dense_features_4/Fare/strided_slice/stack_1:output:04dense_features_4/Fare/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features_4/Fare/strided_sliceљ
%dense_features_4/Fare/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features_4/Fare/Reshape/shape/1я
#dense_features_4/Fare/Reshape/shapePack,dense_features_4/Fare/strided_slice:output:0.dense_features_4/Fare/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features_4/Fare/Reshape/shapeк
dense_features_4/Fare/ReshapeReshapedense_features_4/Cast_2:y:0,dense_features_4/Fare/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
dense_features_4/Fare/ReshapeЌ
dense_features_4/IsAlone/CastCastinputs_isalone*

DstT0*

SrcT0	*'
_output_shapes
:         2
dense_features_4/IsAlone/CastЉ
dense_features_4/IsAlone/ShapeShape!dense_features_4/IsAlone/Cast:y:0*
T0*
_output_shapes
:2 
dense_features_4/IsAlone/Shapeд
,dense_features_4/IsAlone/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,dense_features_4/IsAlone/strided_slice/stackф
.dense_features_4/IsAlone/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_4/IsAlone/strided_slice/stack_1ф
.dense_features_4/IsAlone/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_4/IsAlone/strided_slice/stack_2Э
&dense_features_4/IsAlone/strided_sliceStridedSlice'dense_features_4/IsAlone/Shape:output:05dense_features_4/IsAlone/strided_slice/stack:output:07dense_features_4/IsAlone/strided_slice/stack_1:output:07dense_features_4/IsAlone/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&dense_features_4/IsAlone/strided_sliceќ
(dense_features_4/IsAlone/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(dense_features_4/IsAlone/Reshape/shape/1Ж
&dense_features_4/IsAlone/Reshape/shapePack/dense_features_4/IsAlone/strided_slice:output:01dense_features_4/IsAlone/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&dense_features_4/IsAlone/Reshape/shapeН
 dense_features_4/IsAlone/ReshapeReshape!dense_features_4/IsAlone/Cast:y:0/dense_features_4/IsAlone/Reshape/shape:output:0*
T0*'
_output_shapes
:         2"
 dense_features_4/IsAlone/ReshapeЉ
!dense_features_4/NameLength/ShapeShapedense_features_4/Cast_3:y:0*
T0*
_output_shapes
:2#
!dense_features_4/NameLength/Shapeг
/dense_features_4/NameLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_features_4/NameLength/strided_slice/stack░
1dense_features_4/NameLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features_4/NameLength/strided_slice/stack_1░
1dense_features_4/NameLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features_4/NameLength/strided_slice/stack_2і
)dense_features_4/NameLength/strided_sliceStridedSlice*dense_features_4/NameLength/Shape:output:08dense_features_4/NameLength/strided_slice/stack:output:0:dense_features_4/NameLength/strided_slice/stack_1:output:0:dense_features_4/NameLength/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_features_4/NameLength/strided_sliceю
+dense_features_4/NameLength/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+dense_features_4/NameLength/Reshape/shape/1Ш
)dense_features_4/NameLength/Reshape/shapePack2dense_features_4/NameLength/strided_slice:output:04dense_features_4/NameLength/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)dense_features_4/NameLength/Reshape/shapeп
#dense_features_4/NameLength/ReshapeReshapedense_features_4/Cast_3:y:02dense_features_4/NameLength/Reshape/shape:output:0*
T0*'
_output_shapes
:         2%
#dense_features_4/NameLength/ReshapeЅ
dense_features_4/Pclass/ShapeShapedense_features_4/Cast_4:y:0*
T0*
_output_shapes
:2
dense_features_4/Pclass/Shapeц
+dense_features_4/Pclass/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features_4/Pclass/strided_slice/stackе
-dense_features_4/Pclass/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_4/Pclass/strided_slice/stack_1е
-dense_features_4/Pclass/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_4/Pclass/strided_slice/stack_2Ы
%dense_features_4/Pclass/strided_sliceStridedSlice&dense_features_4/Pclass/Shape:output:04dense_features_4/Pclass/strided_slice/stack:output:06dense_features_4/Pclass/strided_slice/stack_1:output:06dense_features_4/Pclass/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features_4/Pclass/strided_sliceћ
'dense_features_4/Pclass/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features_4/Pclass/Reshape/shape/1Т
%dense_features_4/Pclass/Reshape/shapePack.dense_features_4/Pclass/strided_slice:output:00dense_features_4/Pclass/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features_4/Pclass/Reshape/shape╠
dense_features_4/Pclass/ReshapeReshapedense_features_4/Cast_4:y:0.dense_features_4/Pclass/Reshape/shape:output:0*
T0*'
_output_shapes
:         2!
dense_features_4/Pclass/ReshapeІ
dense_features_4/Sex/CastCast
inputs_sex*

DstT0*

SrcT0	*'
_output_shapes
:         2
dense_features_4/Sex/CastЁ
dense_features_4/Sex/ShapeShapedense_features_4/Sex/Cast:y:0*
T0*
_output_shapes
:2
dense_features_4/Sex/Shapeъ
(dense_features_4/Sex/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features_4/Sex/strided_slice/stackб
*dense_features_4/Sex/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Sex/strided_slice/stack_1б
*dense_features_4/Sex/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Sex/strided_slice/stack_2Я
"dense_features_4/Sex/strided_sliceStridedSlice#dense_features_4/Sex/Shape:output:01dense_features_4/Sex/strided_slice/stack:output:03dense_features_4/Sex/strided_slice/stack_1:output:03dense_features_4/Sex/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features_4/Sex/strided_sliceј
$dense_features_4/Sex/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features_4/Sex/Reshape/shape/1┌
"dense_features_4/Sex/Reshape/shapePack+dense_features_4/Sex/strided_slice:output:0-dense_features_4/Sex/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features_4/Sex/Reshape/shape┼
dense_features_4/Sex/ReshapeReshapedense_features_4/Sex/Cast:y:0+dense_features_4/Sex/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
dense_features_4/Sex/ReshapeЄ
dense_features_4/Title/ShapeShapedense_features_4/Cast_5:y:0*
T0*
_output_shapes
:2
dense_features_4/Title/Shapeб
*dense_features_4/Title/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_4/Title/strided_slice/stackд
,dense_features_4/Title/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_4/Title/strided_slice/stack_1д
,dense_features_4/Title/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_4/Title/strided_slice/stack_2В
$dense_features_4/Title/strided_sliceStridedSlice%dense_features_4/Title/Shape:output:03dense_features_4/Title/strided_slice/stack:output:05dense_features_4/Title/strided_slice/stack_1:output:05dense_features_4/Title/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_4/Title/strided_sliceњ
&dense_features_4/Title/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_4/Title/Reshape/shape/1Р
$dense_features_4/Title/Reshape/shapePack-dense_features_4/Title/strided_slice:output:0/dense_features_4/Title/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_4/Title/Reshape/shape╔
dense_features_4/Title/ReshapeReshapedense_features_4/Cast_5:y:0-dense_features_4/Title/Reshape/shape:output:0*
T0*'
_output_shapes
:         2 
dense_features_4/Title/ReshapeЄ
dense_features_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2
dense_features_4/concat/axisЬ
dense_features_4/concatConcatV2%dense_features_4/Age/Reshape:output:0*dense_features_4/Embarked/Reshape:output:0&dense_features_4/Fare/Reshape:output:0)dense_features_4/IsAlone/Reshape:output:0,dense_features_4/NameLength/Reshape:output:0(dense_features_4/Pclass/Reshape:output:0%dense_features_4/Sex/Reshape:output:0'dense_features_4/Title/Reshape:output:0%dense_features_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
dense_features_4/concatЕ
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	ъ*
dtype02 
dense_20/MatMul/ReadVariableOpЕ
dense_20/MatMulMatMul dense_features_4/concat:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
dense_20/MatMulе
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:ъ*
dtype02!
dense_20/BiasAdd/ReadVariableOpд
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
dense_20/BiasAddt
dense_20/SeluSeludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:         ъ2
dense_20/Seluy
dropout_16/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_16/dropout/Constф
dropout_16/dropout/MulMuldense_20/Selu:activations:0!dropout_16/dropout/Const:output:0*
T0*(
_output_shapes
:         ъ2
dropout_16/dropout/Mul
dropout_16/dropout/ShapeShapedense_20/Selu:activations:0*
T0*
_output_shapes
:2
dropout_16/dropout/Shapeо
/dropout_16/dropout/random_uniform/RandomUniformRandomUniform!dropout_16/dropout/Shape:output:0*
T0*(
_output_shapes
:         ъ*
dtype021
/dropout_16/dropout/random_uniform/RandomUniformІ
!dropout_16/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_16/dropout/GreaterEqual/yв
dropout_16/dropout/GreaterEqualGreaterEqual8dropout_16/dropout/random_uniform/RandomUniform:output:0*dropout_16/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ъ2!
dropout_16/dropout/GreaterEqualА
dropout_16/dropout/CastCast#dropout_16/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ъ2
dropout_16/dropout/CastД
dropout_16/dropout/Mul_1Muldropout_16/dropout/Mul:z:0dropout_16/dropout/Cast:y:0*
T0*(
_output_shapes
:         ъ2
dropout_16/dropout/Mul_1ф
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ъе*
dtype02 
dense_21/MatMul/ReadVariableOpЦ
dense_21/MatMulMatMuldropout_16/dropout/Mul_1:z:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
dense_21/MatMulе
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:е*
dtype02!
dense_21/BiasAdd/ReadVariableOpд
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
dense_21/BiasAdd}
dense_21/SigmoidSigmoiddense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         е2
dense_21/SigmoidЄ
dense_21/mulMuldense_21/BiasAdd:output:0dense_21/Sigmoid:y:0*
T0*(
_output_shapes
:         е2
dense_21/mulw
dense_21/IdentityIdentitydense_21/mul:z:0*
T0*(
_output_shapes
:         е2
dense_21/Identity█
dense_21/IdentityN	IdentityNdense_21/mul:z:0dense_21/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-213226*<
_output_shapes*
(:         е:         е2
dense_21/IdentityNy
dropout_17/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_17/dropout/Constф
dropout_17/dropout/MulMuldense_21/IdentityN:output:0!dropout_17/dropout/Const:output:0*
T0*(
_output_shapes
:         е2
dropout_17/dropout/Mul
dropout_17/dropout/ShapeShapedense_21/IdentityN:output:0*
T0*
_output_shapes
:2
dropout_17/dropout/Shapeо
/dropout_17/dropout/random_uniform/RandomUniformRandomUniform!dropout_17/dropout/Shape:output:0*
T0*(
_output_shapes
:         е*
dtype021
/dropout_17/dropout/random_uniform/RandomUniformІ
!dropout_17/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_17/dropout/GreaterEqual/yв
dropout_17/dropout/GreaterEqualGreaterEqual8dropout_17/dropout/random_uniform/RandomUniform:output:0*dropout_17/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         е2!
dropout_17/dropout/GreaterEqualА
dropout_17/dropout/CastCast#dropout_17/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         е2
dropout_17/dropout/CastД
dropout_17/dropout/Mul_1Muldropout_17/dropout/Mul:z:0dropout_17/dropout/Cast:y:0*
T0*(
_output_shapes
:         е2
dropout_17/dropout/Mul_1ф
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
е▓*
dtype02 
dense_22/MatMul/ReadVariableOpЦ
dense_22/MatMulMatMuldropout_17/dropout/Mul_1:z:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
dense_22/MatMulе
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:▓*
dtype02!
dense_22/BiasAdd/ReadVariableOpд
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
dense_22/BiasAdd}
dense_22/SigmoidSigmoiddense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ▓2
dense_22/SigmoidЄ
dense_22/mulMuldense_22/BiasAdd:output:0dense_22/Sigmoid:y:0*
T0*(
_output_shapes
:         ▓2
dense_22/mulw
dense_22/IdentityIdentitydense_22/mul:z:0*
T0*(
_output_shapes
:         ▓2
dense_22/Identity█
dense_22/IdentityN	IdentityNdense_22/mul:z:0dense_22/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-213246*<
_output_shapes*
(:         ▓:         ▓2
dense_22/IdentityNy
dropout_18/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_18/dropout/Constф
dropout_18/dropout/MulMuldense_22/IdentityN:output:0!dropout_18/dropout/Const:output:0*
T0*(
_output_shapes
:         ▓2
dropout_18/dropout/Mul
dropout_18/dropout/ShapeShapedense_22/IdentityN:output:0*
T0*
_output_shapes
:2
dropout_18/dropout/Shapeо
/dropout_18/dropout/random_uniform/RandomUniformRandomUniform!dropout_18/dropout/Shape:output:0*
T0*(
_output_shapes
:         ▓*
dtype021
/dropout_18/dropout/random_uniform/RandomUniformІ
!dropout_18/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_18/dropout/GreaterEqual/yв
dropout_18/dropout/GreaterEqualGreaterEqual8dropout_18/dropout/random_uniform/RandomUniform:output:0*dropout_18/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ▓2!
dropout_18/dropout/GreaterEqualА
dropout_18/dropout/CastCast#dropout_18/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ▓2
dropout_18/dropout/CastД
dropout_18/dropout/Mul_1Muldropout_18/dropout/Mul:z:0dropout_18/dropout/Cast:y:0*
T0*(
_output_shapes
:         ▓2
dropout_18/dropout/Mul_1ф
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
▓╝*
dtype02 
dense_23/MatMul/ReadVariableOpЦ
dense_23/MatMulMatMuldropout_18/dropout/Mul_1:z:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
dense_23/MatMulе
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:╝*
dtype02!
dense_23/BiasAdd/ReadVariableOpд
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
dense_23/BiasAddt
dense_23/SeluSeludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:         ╝2
dense_23/Seluy
dropout_19/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout_19/dropout/Constф
dropout_19/dropout/MulMuldense_23/Selu:activations:0!dropout_19/dropout/Const:output:0*
T0*(
_output_shapes
:         ╝2
dropout_19/dropout/Mul
dropout_19/dropout/ShapeShapedense_23/Selu:activations:0*
T0*
_output_shapes
:2
dropout_19/dropout/Shapeо
/dropout_19/dropout/random_uniform/RandomUniformRandomUniform!dropout_19/dropout/Shape:output:0*
T0*(
_output_shapes
:         ╝*
dtype021
/dropout_19/dropout/random_uniform/RandomUniformІ
!dropout_19/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2#
!dropout_19/dropout/GreaterEqual/yв
dropout_19/dropout/GreaterEqualGreaterEqual8dropout_19/dropout/random_uniform/RandomUniform:output:0*dropout_19/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╝2!
dropout_19/dropout/GreaterEqualА
dropout_19/dropout/CastCast#dropout_19/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╝2
dropout_19/dropout/CastД
dropout_19/dropout/Mul_1Muldropout_19/dropout/Mul:z:0dropout_19/dropout/Cast:y:0*
T0*(
_output_shapes
:         ╝2
dropout_19/dropout/Mul_1Е
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	╝*
dtype02 
dense_24/MatMul/ReadVariableOpц
dense_24/MatMulMatMuldropout_19/dropout/Mul_1:z:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/MatMulД
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpЦ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/BiasAdd|
dense_24/SigmoidSigmoiddense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_24/Sigmoidh
IdentityIdentitydense_24/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         :::::::::::S O
'
_output_shapes
:         
$
_user_specified_name
inputs/Age:XT
'
_output_shapes
:         
)
_user_specified_nameinputs/Embarked:TP
'
_output_shapes
:         
%
_user_specified_nameinputs/Fare:WS
'
_output_shapes
:         
(
_user_specified_nameinputs/IsAlone:ZV
'
_output_shapes
:         
+
_user_specified_nameinputs/NameLength:VR
'
_output_shapes
:         
'
_user_specified_nameinputs/Pclass:SO
'
_output_shapes
:         
$
_user_specified_name
inputs/Sex:UQ
'
_output_shapes
:         
&
_user_specified_nameinputs/Title:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
■
«
D__inference_dense_21_layer_call_and_return_conditional_losses_212680

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ъе*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:е*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         е2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         е2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:         е2

Identityи
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-212673*<
_output_shapes*
(:         е:         е2
	IdentityNk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:         е2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         ъ:::P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
І
e
F__inference_dropout_18_layer_call_and_return_conditional_losses_213708

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ▓2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ▓*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ▓2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ▓2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ▓2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ▓2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ▓:P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs
═
d
F__inference_dropout_18_layer_call_and_return_conditional_losses_213713

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ▓2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ▓2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ▓:P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs
І
e
F__inference_dropout_19_layer_call_and_return_conditional_losses_213755

inputs
identityѕc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  а?2
dropout/Constt
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         ╝2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeх
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         ╝*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *═╠L>2
dropout/GreaterEqual/y┐
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         ╝2
dropout/GreaterEqualђ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         ╝2
dropout/Cast{
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         ╝2
dropout/Mul_1f
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         ╝2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ╝:P L
(
_output_shapes
:         ╝
 
_user_specified_nameinputs
ю»
ц
"__inference__traced_restore_214070
file_prefix1
-assignvariableop_sequential_4_dense_20_kernel1
-assignvariableop_1_sequential_4_dense_20_bias3
/assignvariableop_2_sequential_4_dense_21_kernel1
-assignvariableop_3_sequential_4_dense_21_bias3
/assignvariableop_4_sequential_4_dense_22_kernel1
-assignvariableop_5_sequential_4_dense_22_bias3
/assignvariableop_6_sequential_4_dense_23_kernel1
-assignvariableop_7_sequential_4_dense_23_bias3
/assignvariableop_8_sequential_4_dense_24_kernel1
-assignvariableop_9_sequential_4_dense_24_bias!
assignvariableop_10_adam_iter#
assignvariableop_11_adam_beta_1#
assignvariableop_12_adam_beta_2"
assignvariableop_13_adam_decay*
&assignvariableop_14_adam_learning_rate
assignvariableop_15_total
assignvariableop_16_count
assignvariableop_17_total_1
assignvariableop_18_count_1;
7assignvariableop_19_adam_sequential_4_dense_20_kernel_m9
5assignvariableop_20_adam_sequential_4_dense_20_bias_m;
7assignvariableop_21_adam_sequential_4_dense_21_kernel_m9
5assignvariableop_22_adam_sequential_4_dense_21_bias_m;
7assignvariableop_23_adam_sequential_4_dense_22_kernel_m9
5assignvariableop_24_adam_sequential_4_dense_22_bias_m;
7assignvariableop_25_adam_sequential_4_dense_23_kernel_m9
5assignvariableop_26_adam_sequential_4_dense_23_bias_m;
7assignvariableop_27_adam_sequential_4_dense_24_kernel_m9
5assignvariableop_28_adam_sequential_4_dense_24_bias_m;
7assignvariableop_29_adam_sequential_4_dense_20_kernel_v9
5assignvariableop_30_adam_sequential_4_dense_20_bias_v;
7assignvariableop_31_adam_sequential_4_dense_21_kernel_v9
5assignvariableop_32_adam_sequential_4_dense_21_bias_v;
7assignvariableop_33_adam_sequential_4_dense_22_kernel_v9
5assignvariableop_34_adam_sequential_4_dense_22_bias_v;
7assignvariableop_35_adam_sequential_4_dense_23_kernel_v9
5assignvariableop_36_adam_sequential_4_dense_23_bias_v;
7assignvariableop_37_adam_sequential_4_dense_24_kernel_v9
5assignvariableop_38_adam_sequential_4_dense_24_bias_v
identity_40ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_21бAssignVariableOp_22бAssignVariableOp_23бAssignVariableOp_24бAssignVariableOp_25бAssignVariableOp_26бAssignVariableOp_27бAssignVariableOp_28бAssignVariableOp_29бAssignVariableOp_3бAssignVariableOp_30бAssignVariableOp_31бAssignVariableOp_32бAssignVariableOp_33бAssignVariableOp_34бAssignVariableOp_35бAssignVariableOp_36бAssignVariableOp_37бAssignVariableOp_38бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1У
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*З
valueЖBу'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names▄
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesы
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*▓
_output_shapesЪ
ю:::::::::::::::::::::::::::::::::::::::*5
dtypes+
)2'	2
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЮ
AssignVariableOpAssignVariableOp-assignvariableop_sequential_4_dense_20_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Б
AssignVariableOp_1AssignVariableOp-assignvariableop_1_sequential_4_dense_20_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Ц
AssignVariableOp_2AssignVariableOp/assignvariableop_2_sequential_4_dense_21_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Б
AssignVariableOp_3AssignVariableOp-assignvariableop_3_sequential_4_dense_21_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4Ц
AssignVariableOp_4AssignVariableOp/assignvariableop_4_sequential_4_dense_22_kernelIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5Б
AssignVariableOp_5AssignVariableOp-assignvariableop_5_sequential_4_dense_22_biasIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6Ц
AssignVariableOp_6AssignVariableOp/assignvariableop_6_sequential_4_dense_23_kernelIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7Б
AssignVariableOp_7AssignVariableOp-assignvariableop_7_sequential_4_dense_23_biasIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8Ц
AssignVariableOp_8AssignVariableOp/assignvariableop_8_sequential_4_dense_24_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9Б
AssignVariableOp_9AssignVariableOp-assignvariableop_9_sequential_4_dense_24_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0	*
_output_shapes
:2
Identity_10ќ
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11ў
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12ў
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13Ќ
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14Ъ
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15њ
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16њ
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17ћ
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18ћ
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19░
AssignVariableOp_19AssignVariableOp7assignvariableop_19_adam_sequential_4_dense_20_kernel_mIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20«
AssignVariableOp_20AssignVariableOp5assignvariableop_20_adam_sequential_4_dense_20_bias_mIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21░
AssignVariableOp_21AssignVariableOp7assignvariableop_21_adam_sequential_4_dense_21_kernel_mIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22«
AssignVariableOp_22AssignVariableOp5assignvariableop_22_adam_sequential_4_dense_21_bias_mIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23░
AssignVariableOp_23AssignVariableOp7assignvariableop_23_adam_sequential_4_dense_22_kernel_mIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24«
AssignVariableOp_24AssignVariableOp5assignvariableop_24_adam_sequential_4_dense_22_bias_mIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25░
AssignVariableOp_25AssignVariableOp7assignvariableop_25_adam_sequential_4_dense_23_kernel_mIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26«
AssignVariableOp_26AssignVariableOp5assignvariableop_26_adam_sequential_4_dense_23_bias_mIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27░
AssignVariableOp_27AssignVariableOp7assignvariableop_27_adam_sequential_4_dense_24_kernel_mIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28«
AssignVariableOp_28AssignVariableOp5assignvariableop_28_adam_sequential_4_dense_24_bias_mIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29░
AssignVariableOp_29AssignVariableOp7assignvariableop_29_adam_sequential_4_dense_20_kernel_vIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30«
AssignVariableOp_30AssignVariableOp5assignvariableop_30_adam_sequential_4_dense_20_bias_vIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31░
AssignVariableOp_31AssignVariableOp7assignvariableop_31_adam_sequential_4_dense_21_kernel_vIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32«
AssignVariableOp_32AssignVariableOp5assignvariableop_32_adam_sequential_4_dense_21_bias_vIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33░
AssignVariableOp_33AssignVariableOp7assignvariableop_33_adam_sequential_4_dense_22_kernel_vIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34«
AssignVariableOp_34AssignVariableOp5assignvariableop_34_adam_sequential_4_dense_22_bias_vIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35░
AssignVariableOp_35AssignVariableOp7assignvariableop_35_adam_sequential_4_dense_23_kernel_vIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36«
AssignVariableOp_36AssignVariableOp5assignvariableop_36_adam_sequential_4_dense_23_bias_vIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37░
AssignVariableOp_37AssignVariableOp7assignvariableop_37_adam_sequential_4_dense_24_kernel_vIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38«
AssignVariableOp_38AssignVariableOp5assignvariableop_38_adam_sequential_4_dense_24_bias_vIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38е
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesћ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices─
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpИ
Identity_39Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_39┼
Identity_40IdentityIdentity_39:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_40"#
identity_40Identity_40:output:0*│
_input_shapesА
ъ: :::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: 
ч
G
+__inference_dropout_18_layer_call_fn_213723

inputs
identityд
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ▓* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_2127752
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         ▓2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ▓:P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs
■
«
D__inference_dense_22_layer_call_and_return_conditional_losses_213687

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource

identity_1ѕЈ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
е▓*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
MatMulЇ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:▓*
dtype02
BiasAdd/ReadVariableOpѓ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2	
BiasAddb
SigmoidSigmoidBiasAdd:output:0*
T0*(
_output_shapes
:         ▓2	
Sigmoidc
mulMulBiasAdd:output:0Sigmoid:y:0*
T0*(
_output_shapes
:         ▓2
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:         ▓2

Identityи
	IdentityN	IdentityNmul:z:0BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-213680*<
_output_shapes*
(:         ▓:         ▓2
	IdentityNk

Identity_1IdentityIdentityN:output:0*
T0*(
_output_shapes
:         ▓2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:         е:::P L
(
_output_shapes
:         е
 
_user_specified_nameinputs:

_output_shapes
: :

_output_shapes
: 
═
d
F__inference_dropout_16_layer_call_and_return_conditional_losses_212651

inputs

identity_1[
IdentityIdentityinputs*
T0*(
_output_shapes
:         ъ2

Identityj

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         ъ2

Identity_1"!

identity_1Identity_1:output:0*'
_input_shapes
:         ъ:P L
(
_output_shapes
:         ъ
 
_user_specified_nameinputs
Й
╠
-__inference_sequential_4_layer_call_fn_213079
age
embarked
fare
isalone	

namelength

pclass
sex		
title
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityѕбStatefulPartitionedCall■
StatefulPartitionedCallStatefulPartitionedCallageembarkedfareisalone
namelengthpclasssextitleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_2130562
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:         

_user_specified_nameAge:QM
'
_output_shapes
:         
"
_user_specified_name
Embarked:MI
'
_output_shapes
:         

_user_specified_nameFare:PL
'
_output_shapes
:         
!
_user_specified_name	IsAlone:SO
'
_output_shapes
:         
$
_user_specified_name
NameLength:OK
'
_output_shapes
:         
 
_user_specified_namePclass:LH
'
_output_shapes
:         

_user_specified_nameSex:NJ
'
_output_shapes
:         

_user_specified_nameTitle:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
Є
d
+__inference_dropout_18_layer_call_fn_213718

inputs
identityѕбStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*(
_output_shapes
:         ▓* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_2127702
StatefulPartitionedCallЈ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:         ▓2

Identity"
identityIdentity:output:0*'
_input_shapes
:         ▓22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         ▓
 
_user_specified_nameinputs
ч_
њ
__inference__traced_save_213941
file_prefix;
7savev2_sequential_4_dense_20_kernel_read_readvariableop9
5savev2_sequential_4_dense_20_bias_read_readvariableop;
7savev2_sequential_4_dense_21_kernel_read_readvariableop9
5savev2_sequential_4_dense_21_bias_read_readvariableop;
7savev2_sequential_4_dense_22_kernel_read_readvariableop9
5savev2_sequential_4_dense_22_bias_read_readvariableop;
7savev2_sequential_4_dense_23_kernel_read_readvariableop9
5savev2_sequential_4_dense_23_bias_read_readvariableop;
7savev2_sequential_4_dense_24_kernel_read_readvariableop9
5savev2_sequential_4_dense_24_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableopB
>savev2_adam_sequential_4_dense_20_kernel_m_read_readvariableop@
<savev2_adam_sequential_4_dense_20_bias_m_read_readvariableopB
>savev2_adam_sequential_4_dense_21_kernel_m_read_readvariableop@
<savev2_adam_sequential_4_dense_21_bias_m_read_readvariableopB
>savev2_adam_sequential_4_dense_22_kernel_m_read_readvariableop@
<savev2_adam_sequential_4_dense_22_bias_m_read_readvariableopB
>savev2_adam_sequential_4_dense_23_kernel_m_read_readvariableop@
<savev2_adam_sequential_4_dense_23_bias_m_read_readvariableopB
>savev2_adam_sequential_4_dense_24_kernel_m_read_readvariableop@
<savev2_adam_sequential_4_dense_24_bias_m_read_readvariableopB
>savev2_adam_sequential_4_dense_20_kernel_v_read_readvariableop@
<savev2_adam_sequential_4_dense_20_bias_v_read_readvariableopB
>savev2_adam_sequential_4_dense_21_kernel_v_read_readvariableop@
<savev2_adam_sequential_4_dense_21_bias_v_read_readvariableopB
>savev2_adam_sequential_4_dense_22_kernel_v_read_readvariableop@
<savev2_adam_sequential_4_dense_22_bias_v_read_readvariableopB
>savev2_adam_sequential_4_dense_23_kernel_v_read_readvariableop@
<savev2_adam_sequential_4_dense_23_bias_v_read_readvariableopB
>savev2_adam_sequential_4_dense_24_kernel_v_read_readvariableop@
<savev2_adam_sequential_4_dense_24_bias_v_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1Ј
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
ConstЇ
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_698459220f7e421789095f08f09b3369/part2	
Const_1І
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardд
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameР
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*З
valueЖBу'B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesо
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:'*
dtype0*a
valueXBV'B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesк
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:07savev2_sequential_4_dense_20_kernel_read_readvariableop5savev2_sequential_4_dense_20_bias_read_readvariableop7savev2_sequential_4_dense_21_kernel_read_readvariableop5savev2_sequential_4_dense_21_bias_read_readvariableop7savev2_sequential_4_dense_22_kernel_read_readvariableop5savev2_sequential_4_dense_22_bias_read_readvariableop7savev2_sequential_4_dense_23_kernel_read_readvariableop5savev2_sequential_4_dense_23_bias_read_readvariableop7savev2_sequential_4_dense_24_kernel_read_readvariableop5savev2_sequential_4_dense_24_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop>savev2_adam_sequential_4_dense_20_kernel_m_read_readvariableop<savev2_adam_sequential_4_dense_20_bias_m_read_readvariableop>savev2_adam_sequential_4_dense_21_kernel_m_read_readvariableop<savev2_adam_sequential_4_dense_21_bias_m_read_readvariableop>savev2_adam_sequential_4_dense_22_kernel_m_read_readvariableop<savev2_adam_sequential_4_dense_22_bias_m_read_readvariableop>savev2_adam_sequential_4_dense_23_kernel_m_read_readvariableop<savev2_adam_sequential_4_dense_23_bias_m_read_readvariableop>savev2_adam_sequential_4_dense_24_kernel_m_read_readvariableop<savev2_adam_sequential_4_dense_24_bias_m_read_readvariableop>savev2_adam_sequential_4_dense_20_kernel_v_read_readvariableop<savev2_adam_sequential_4_dense_20_bias_v_read_readvariableop>savev2_adam_sequential_4_dense_21_kernel_v_read_readvariableop<savev2_adam_sequential_4_dense_21_bias_v_read_readvariableop>savev2_adam_sequential_4_dense_22_kernel_v_read_readvariableop<savev2_adam_sequential_4_dense_22_bias_v_read_readvariableop>savev2_adam_sequential_4_dense_23_kernel_v_read_readvariableop<savev2_adam_sequential_4_dense_23_bias_v_read_readvariableop>savev2_adam_sequential_4_dense_24_kernel_v_read_readvariableop<savev2_adam_sequential_4_dense_24_bias_v_read_readvariableop"/device:CPU:0*
_output_shapes
 *5
dtypes+
)2'	2
SaveV2Ѓ
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardг
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1б
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesј
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices¤
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1с
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesг
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityЂ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*┐
_input_shapesГ
ф: :	ъ:ъ:
ъе:е:
е▓:▓:
▓╝:╝:	╝:: : : : : : : : : :	ъ:ъ:
ъе:е:
е▓:▓:
▓╝:╝:	╝::	ъ:ъ:
ъе:е:
е▓:▓:
▓╝:╝:	╝:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	ъ:!

_output_shapes	
:ъ:&"
 
_output_shapes
:
ъе:!

_output_shapes	
:е:&"
 
_output_shapes
:
е▓:!

_output_shapes	
:▓:&"
 
_output_shapes
:
▓╝:!

_output_shapes	
:╝:%	!

_output_shapes
:	╝: 


_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :
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
: :

_output_shapes
: :%!

_output_shapes
:	ъ:!

_output_shapes	
:ъ:&"
 
_output_shapes
:
ъе:!

_output_shapes	
:е:&"
 
_output_shapes
:
е▓:!

_output_shapes	
:▓:&"
 
_output_shapes
:
▓╝:!

_output_shapes	
:╝:%!

_output_shapes
:	╝: 

_output_shapes
::%!

_output_shapes
:	ъ:!

_output_shapes	
:ъ:& "
 
_output_shapes
:
ъе:!!

_output_shapes	
:е:&""
 
_output_shapes
:
е▓:!#

_output_shapes	
:▓:&$"
 
_output_shapes
:
▓╝:!%

_output_shapes	
:╝:%&!

_output_shapes
:	╝: '

_output_shapes
::(

_output_shapes
: 
Т
ё
-__inference_sequential_4_layer_call_fn_213481

inputs_age
inputs_embarked
inputs_fare
inputs_isalone	
inputs_namelength
inputs_pclass

inputs_sex	
inputs_title
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCall
inputs_ageinputs_embarkedinputs_fareinputs_isaloneinputs_namelengthinputs_pclass
inputs_sexinputs_titleunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2		*
Tout
2*'
_output_shapes
:         *,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU2*0J 8*Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_2130562
StatefulPartitionedCallј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:S O
'
_output_shapes
:         
$
_user_specified_name
inputs/Age:XT
'
_output_shapes
:         
)
_user_specified_nameinputs/Embarked:TP
'
_output_shapes
:         
%
_user_specified_nameinputs/Fare:WS
'
_output_shapes
:         
(
_user_specified_nameinputs/IsAlone:ZV
'
_output_shapes
:         
+
_user_specified_nameinputs/NameLength:VR
'
_output_shapes
:         
'
_user_specified_nameinputs/Pclass:SO
'
_output_shapes
:         
$
_user_specified_name
inputs/Sex:UQ
'
_output_shapes
:         
&
_user_specified_nameinputs/Title:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
ќ8
║
H__inference_sequential_4_layer_call_and_return_conditional_losses_212920
age
embarked
fare
isalone	

namelength

pclass
sex		
title
dense_20_212890
dense_20_212892
dense_21_212896
dense_21_212898
dense_22_212902
dense_22_212904
dense_23_212908
dense_23_212910
dense_24_212914
dense_24_212916
identityѕб dense_20/StatefulPartitionedCallб dense_21/StatefulPartitionedCallб dense_22/StatefulPartitionedCallб dense_23/StatefulPartitionedCallб dense_24/StatefulPartitionedCall|
dense_features_4/CastCastage*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/CastЁ
dense_features_4/Cast_1Castembarked*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_1Ђ
dense_features_4/Cast_2Castfare*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_2Є
dense_features_4/Cast_3Cast
namelength*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_3Ѓ
dense_features_4/Cast_4Castpclass*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_4ѓ
dense_features_4/Cast_5Casttitle*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_5є
 dense_features_4/PartitionedCallPartitionedCalldense_features_4/Cast:y:0dense_features_4/Cast_1:y:0dense_features_4/Cast_2:y:0isalonedense_features_4/Cast_3:y:0dense_features_4/Cast_4:y:0sexdense_features_4/Cast_5:y:0*
Tin

2		*
Tout
2*'
_output_shapes
:         * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*U
fPRN
L__inference_dense_features_4_layer_call_and_return_conditional_losses_2125922"
 dense_features_4/PartitionedCallЎ
 dense_20/StatefulPartitionedCallStatefulPartitionedCall)dense_features_4/PartitionedCall:output:0dense_20_212890dense_20_212892*
Tin
2*
Tout
2*(
_output_shapes
:         ъ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_20_layer_call_and_return_conditional_losses_2126182"
 dense_20/StatefulPartitionedCall▀
dropout_16/PartitionedCallPartitionedCall)dense_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ъ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_16_layer_call_and_return_conditional_losses_2126512
dropout_16/PartitionedCallЊ
 dense_21/StatefulPartitionedCallStatefulPartitionedCall#dropout_16/PartitionedCall:output:0dense_21_212896dense_21_212898*
Tin
2*
Tout
2*(
_output_shapes
:         е*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_21_layer_call_and_return_conditional_losses_2126802"
 dense_21/StatefulPartitionedCall▀
dropout_17/PartitionedCallPartitionedCall)dense_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         е* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_17_layer_call_and_return_conditional_losses_2127132
dropout_17/PartitionedCallЊ
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#dropout_17/PartitionedCall:output:0dense_22_212902dense_22_212904*
Tin
2*
Tout
2*(
_output_shapes
:         ▓*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_22_layer_call_and_return_conditional_losses_2127422"
 dense_22/StatefulPartitionedCall▀
dropout_18/PartitionedCallPartitionedCall)dense_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ▓* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_18_layer_call_and_return_conditional_losses_2127752
dropout_18/PartitionedCallЊ
 dense_23/StatefulPartitionedCallStatefulPartitionedCall#dropout_18/PartitionedCall:output:0dense_23_212908dense_23_212910*
Tin
2*
Tout
2*(
_output_shapes
:         ╝*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_23_layer_call_and_return_conditional_losses_2127992"
 dense_23/StatefulPartitionedCall▀
dropout_19/PartitionedCallPartitionedCall)dense_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*(
_output_shapes
:         ╝* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU2*0J 8*O
fJRH
F__inference_dropout_19_layer_call_and_return_conditional_losses_2128322
dropout_19/PartitionedCallњ
 dense_24/StatefulPartitionedCallStatefulPartitionedCall#dropout_19/PartitionedCall:output:0dense_24_212914dense_24_212916*
Tin
2*
Tout
2*'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU2*0J 8*M
fHRF
D__inference_dense_24_layer_call_and_return_conditional_losses_2128562"
 dense_24/StatefulPartitionedCallг
IdentityIdentity)dense_24/StatefulPartitionedCall:output:0!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall!^dense_24/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         ::::::::::2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall2D
 dense_24/StatefulPartitionedCall dense_24/StatefulPartitionedCall:L H
'
_output_shapes
:         

_user_specified_nameAge:QM
'
_output_shapes
:         
"
_user_specified_name
Embarked:MI
'
_output_shapes
:         

_user_specified_nameFare:PL
'
_output_shapes
:         
!
_user_specified_name	IsAlone:SO
'
_output_shapes
:         
$
_user_specified_name
NameLength:OK
'
_output_shapes
:         
 
_user_specified_namePclass:LH
'
_output_shapes
:         

_user_specified_nameSex:NJ
'
_output_shapes
:         

_user_specified_nameTitle:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
МГ
И
H__inference_sequential_4_layer_call_and_return_conditional_losses_213417

inputs_age
inputs_embarked
inputs_fare
inputs_isalone	
inputs_namelength
inputs_pclass

inputs_sex	
inputs_title+
'dense_20_matmul_readvariableop_resource,
(dense_20_biasadd_readvariableop_resource+
'dense_21_matmul_readvariableop_resource,
(dense_21_biasadd_readvariableop_resource+
'dense_22_matmul_readvariableop_resource,
(dense_22_biasadd_readvariableop_resource+
'dense_23_matmul_readvariableop_resource,
(dense_23_biasadd_readvariableop_resource+
'dense_24_matmul_readvariableop_resource,
(dense_24_biasadd_readvariableop_resource
identityѕЃ
dense_features_4/CastCast
inputs_age*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Castї
dense_features_4/Cast_1Castinputs_embarked*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_1ѕ
dense_features_4/Cast_2Castinputs_fare*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_2ј
dense_features_4/Cast_3Castinputs_namelength*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_3і
dense_features_4/Cast_4Castinputs_pclass*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_4Ѕ
dense_features_4/Cast_5Castinputs_title*

DstT0*

SrcT0*'
_output_shapes
:         2
dense_features_4/Cast_5Ђ
dense_features_4/Age/ShapeShapedense_features_4/Cast:y:0*
T0*
_output_shapes
:2
dense_features_4/Age/Shapeъ
(dense_features_4/Age/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features_4/Age/strided_slice/stackб
*dense_features_4/Age/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Age/strided_slice/stack_1б
*dense_features_4/Age/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Age/strided_slice/stack_2Я
"dense_features_4/Age/strided_sliceStridedSlice#dense_features_4/Age/Shape:output:01dense_features_4/Age/strided_slice/stack:output:03dense_features_4/Age/strided_slice/stack_1:output:03dense_features_4/Age/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features_4/Age/strided_sliceј
$dense_features_4/Age/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features_4/Age/Reshape/shape/1┌
"dense_features_4/Age/Reshape/shapePack+dense_features_4/Age/strided_slice:output:0-dense_features_4/Age/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features_4/Age/Reshape/shape┴
dense_features_4/Age/ReshapeReshapedense_features_4/Cast:y:0+dense_features_4/Age/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
dense_features_4/Age/ReshapeЇ
dense_features_4/Embarked/ShapeShapedense_features_4/Cast_1:y:0*
T0*
_output_shapes
:2!
dense_features_4/Embarked/Shapeе
-dense_features_4/Embarked/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense_features_4/Embarked/strided_slice/stackг
/dense_features_4/Embarked/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features_4/Embarked/strided_slice/stack_1г
/dense_features_4/Embarked/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense_features_4/Embarked/strided_slice/stack_2■
'dense_features_4/Embarked/strided_sliceStridedSlice(dense_features_4/Embarked/Shape:output:06dense_features_4/Embarked/strided_slice/stack:output:08dense_features_4/Embarked/strided_slice/stack_1:output:08dense_features_4/Embarked/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense_features_4/Embarked/strided_sliceў
)dense_features_4/Embarked/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2+
)dense_features_4/Embarked/Reshape/shape/1Ь
'dense_features_4/Embarked/Reshape/shapePack0dense_features_4/Embarked/strided_slice:output:02dense_features_4/Embarked/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2)
'dense_features_4/Embarked/Reshape/shapeм
!dense_features_4/Embarked/ReshapeReshapedense_features_4/Cast_1:y:00dense_features_4/Embarked/Reshape/shape:output:0*
T0*'
_output_shapes
:         2#
!dense_features_4/Embarked/ReshapeЁ
dense_features_4/Fare/ShapeShapedense_features_4/Cast_2:y:0*
T0*
_output_shapes
:2
dense_features_4/Fare/Shapeа
)dense_features_4/Fare/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)dense_features_4/Fare/strided_slice/stackц
+dense_features_4/Fare/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_4/Fare/strided_slice/stack_1ц
+dense_features_4/Fare/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+dense_features_4/Fare/strided_slice/stack_2Т
#dense_features_4/Fare/strided_sliceStridedSlice$dense_features_4/Fare/Shape:output:02dense_features_4/Fare/strided_slice/stack:output:04dense_features_4/Fare/strided_slice/stack_1:output:04dense_features_4/Fare/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#dense_features_4/Fare/strided_sliceљ
%dense_features_4/Fare/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%dense_features_4/Fare/Reshape/shape/1я
#dense_features_4/Fare/Reshape/shapePack,dense_features_4/Fare/strided_slice:output:0.dense_features_4/Fare/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2%
#dense_features_4/Fare/Reshape/shapeк
dense_features_4/Fare/ReshapeReshapedense_features_4/Cast_2:y:0,dense_features_4/Fare/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
dense_features_4/Fare/ReshapeЌ
dense_features_4/IsAlone/CastCastinputs_isalone*

DstT0*

SrcT0	*'
_output_shapes
:         2
dense_features_4/IsAlone/CastЉ
dense_features_4/IsAlone/ShapeShape!dense_features_4/IsAlone/Cast:y:0*
T0*
_output_shapes
:2 
dense_features_4/IsAlone/Shapeд
,dense_features_4/IsAlone/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,dense_features_4/IsAlone/strided_slice/stackф
.dense_features_4/IsAlone/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_4/IsAlone/strided_slice/stack_1ф
.dense_features_4/IsAlone/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.dense_features_4/IsAlone/strided_slice/stack_2Э
&dense_features_4/IsAlone/strided_sliceStridedSlice'dense_features_4/IsAlone/Shape:output:05dense_features_4/IsAlone/strided_slice/stack:output:07dense_features_4/IsAlone/strided_slice/stack_1:output:07dense_features_4/IsAlone/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&dense_features_4/IsAlone/strided_sliceќ
(dense_features_4/IsAlone/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2*
(dense_features_4/IsAlone/Reshape/shape/1Ж
&dense_features_4/IsAlone/Reshape/shapePack/dense_features_4/IsAlone/strided_slice:output:01dense_features_4/IsAlone/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2(
&dense_features_4/IsAlone/Reshape/shapeН
 dense_features_4/IsAlone/ReshapeReshape!dense_features_4/IsAlone/Cast:y:0/dense_features_4/IsAlone/Reshape/shape:output:0*
T0*'
_output_shapes
:         2"
 dense_features_4/IsAlone/ReshapeЉ
!dense_features_4/NameLength/ShapeShapedense_features_4/Cast_3:y:0*
T0*
_output_shapes
:2#
!dense_features_4/NameLength/Shapeг
/dense_features_4/NameLength/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_features_4/NameLength/strided_slice/stack░
1dense_features_4/NameLength/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features_4/NameLength/strided_slice/stack_1░
1dense_features_4/NameLength/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_features_4/NameLength/strided_slice/stack_2і
)dense_features_4/NameLength/strided_sliceStridedSlice*dense_features_4/NameLength/Shape:output:08dense_features_4/NameLength/strided_slice/stack:output:0:dense_features_4/NameLength/strided_slice/stack_1:output:0:dense_features_4/NameLength/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_features_4/NameLength/strided_sliceю
+dense_features_4/NameLength/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+dense_features_4/NameLength/Reshape/shape/1Ш
)dense_features_4/NameLength/Reshape/shapePack2dense_features_4/NameLength/strided_slice:output:04dense_features_4/NameLength/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)dense_features_4/NameLength/Reshape/shapeп
#dense_features_4/NameLength/ReshapeReshapedense_features_4/Cast_3:y:02dense_features_4/NameLength/Reshape/shape:output:0*
T0*'
_output_shapes
:         2%
#dense_features_4/NameLength/ReshapeЅ
dense_features_4/Pclass/ShapeShapedense_features_4/Cast_4:y:0*
T0*
_output_shapes
:2
dense_features_4/Pclass/Shapeц
+dense_features_4/Pclass/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+dense_features_4/Pclass/strided_slice/stackе
-dense_features_4/Pclass/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_4/Pclass/strided_slice/stack_1е
-dense_features_4/Pclass/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-dense_features_4/Pclass/strided_slice/stack_2Ы
%dense_features_4/Pclass/strided_sliceStridedSlice&dense_features_4/Pclass/Shape:output:04dense_features_4/Pclass/strided_slice/stack:output:06dense_features_4/Pclass/strided_slice/stack_1:output:06dense_features_4/Pclass/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%dense_features_4/Pclass/strided_sliceћ
'dense_features_4/Pclass/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'dense_features_4/Pclass/Reshape/shape/1Т
%dense_features_4/Pclass/Reshape/shapePack.dense_features_4/Pclass/strided_slice:output:00dense_features_4/Pclass/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%dense_features_4/Pclass/Reshape/shape╠
dense_features_4/Pclass/ReshapeReshapedense_features_4/Cast_4:y:0.dense_features_4/Pclass/Reshape/shape:output:0*
T0*'
_output_shapes
:         2!
dense_features_4/Pclass/ReshapeІ
dense_features_4/Sex/CastCast
inputs_sex*

DstT0*

SrcT0	*'
_output_shapes
:         2
dense_features_4/Sex/CastЁ
dense_features_4/Sex/ShapeShapedense_features_4/Sex/Cast:y:0*
T0*
_output_shapes
:2
dense_features_4/Sex/Shapeъ
(dense_features_4/Sex/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(dense_features_4/Sex/strided_slice/stackб
*dense_features_4/Sex/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Sex/strided_slice/stack_1б
*dense_features_4/Sex/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*dense_features_4/Sex/strided_slice/stack_2Я
"dense_features_4/Sex/strided_sliceStridedSlice#dense_features_4/Sex/Shape:output:01dense_features_4/Sex/strided_slice/stack:output:03dense_features_4/Sex/strided_slice/stack_1:output:03dense_features_4/Sex/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"dense_features_4/Sex/strided_sliceј
$dense_features_4/Sex/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2&
$dense_features_4/Sex/Reshape/shape/1┌
"dense_features_4/Sex/Reshape/shapePack+dense_features_4/Sex/strided_slice:output:0-dense_features_4/Sex/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2$
"dense_features_4/Sex/Reshape/shape┼
dense_features_4/Sex/ReshapeReshapedense_features_4/Sex/Cast:y:0+dense_features_4/Sex/Reshape/shape:output:0*
T0*'
_output_shapes
:         2
dense_features_4/Sex/ReshapeЄ
dense_features_4/Title/ShapeShapedense_features_4/Cast_5:y:0*
T0*
_output_shapes
:2
dense_features_4/Title/Shapeб
*dense_features_4/Title/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*dense_features_4/Title/strided_slice/stackд
,dense_features_4/Title/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_4/Title/strided_slice/stack_1д
,dense_features_4/Title/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,dense_features_4/Title/strided_slice/stack_2В
$dense_features_4/Title/strided_sliceStridedSlice%dense_features_4/Title/Shape:output:03dense_features_4/Title/strided_slice/stack:output:05dense_features_4/Title/strided_slice/stack_1:output:05dense_features_4/Title/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$dense_features_4/Title/strided_sliceњ
&dense_features_4/Title/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&dense_features_4/Title/Reshape/shape/1Р
$dense_features_4/Title/Reshape/shapePack-dense_features_4/Title/strided_slice:output:0/dense_features_4/Title/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2&
$dense_features_4/Title/Reshape/shape╔
dense_features_4/Title/ReshapeReshapedense_features_4/Cast_5:y:0-dense_features_4/Title/Reshape/shape:output:0*
T0*'
_output_shapes
:         2 
dense_features_4/Title/ReshapeЄ
dense_features_4/concat/axisConst*
_output_shapes
: *
dtype0*
valueB :
         2
dense_features_4/concat/axisЬ
dense_features_4/concatConcatV2%dense_features_4/Age/Reshape:output:0*dense_features_4/Embarked/Reshape:output:0&dense_features_4/Fare/Reshape:output:0)dense_features_4/IsAlone/Reshape:output:0,dense_features_4/NameLength/Reshape:output:0(dense_features_4/Pclass/Reshape:output:0%dense_features_4/Sex/Reshape:output:0'dense_features_4/Title/Reshape:output:0%dense_features_4/concat/axis:output:0*
N*
T0*'
_output_shapes
:         2
dense_features_4/concatЕ
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	ъ*
dtype02 
dense_20/MatMul/ReadVariableOpЕ
dense_20/MatMulMatMul dense_features_4/concat:output:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
dense_20/MatMulе
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:ъ*
dtype02!
dense_20/BiasAdd/ReadVariableOpд
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ъ2
dense_20/BiasAddt
dense_20/SeluSeludense_20/BiasAdd:output:0*
T0*(
_output_shapes
:         ъ2
dense_20/Seluє
dropout_16/IdentityIdentitydense_20/Selu:activations:0*
T0*(
_output_shapes
:         ъ2
dropout_16/Identityф
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
ъе*
dtype02 
dense_21/MatMul/ReadVariableOpЦ
dense_21/MatMulMatMuldropout_16/Identity:output:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
dense_21/MatMulе
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:е*
dtype02!
dense_21/BiasAdd/ReadVariableOpд
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         е2
dense_21/BiasAdd}
dense_21/SigmoidSigmoiddense_21/BiasAdd:output:0*
T0*(
_output_shapes
:         е2
dense_21/SigmoidЄ
dense_21/mulMuldense_21/BiasAdd:output:0dense_21/Sigmoid:y:0*
T0*(
_output_shapes
:         е2
dense_21/mulw
dense_21/IdentityIdentitydense_21/mul:z:0*
T0*(
_output_shapes
:         е2
dense_21/Identity█
dense_21/IdentityN	IdentityNdense_21/mul:z:0dense_21/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-213381*<
_output_shapes*
(:         е:         е2
dense_21/IdentityNє
dropout_17/IdentityIdentitydense_21/IdentityN:output:0*
T0*(
_output_shapes
:         е2
dropout_17/Identityф
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
е▓*
dtype02 
dense_22/MatMul/ReadVariableOpЦ
dense_22/MatMulMatMuldropout_17/Identity:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
dense_22/MatMulе
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:▓*
dtype02!
dense_22/BiasAdd/ReadVariableOpд
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ▓2
dense_22/BiasAdd}
dense_22/SigmoidSigmoiddense_22/BiasAdd:output:0*
T0*(
_output_shapes
:         ▓2
dense_22/SigmoidЄ
dense_22/mulMuldense_22/BiasAdd:output:0dense_22/Sigmoid:y:0*
T0*(
_output_shapes
:         ▓2
dense_22/mulw
dense_22/IdentityIdentitydense_22/mul:z:0*
T0*(
_output_shapes
:         ▓2
dense_22/Identity█
dense_22/IdentityN	IdentityNdense_22/mul:z:0dense_22/BiasAdd:output:0*
T
2*,
_gradient_op_typeCustomGradient-213394*<
_output_shapes*
(:         ▓:         ▓2
dense_22/IdentityNє
dropout_18/IdentityIdentitydense_22/IdentityN:output:0*
T0*(
_output_shapes
:         ▓2
dropout_18/Identityф
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource* 
_output_shapes
:
▓╝*
dtype02 
dense_23/MatMul/ReadVariableOpЦ
dense_23/MatMulMatMuldropout_18/Identity:output:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
dense_23/MatMulе
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes	
:╝*
dtype02!
dense_23/BiasAdd/ReadVariableOpд
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ╝2
dense_23/BiasAddt
dense_23/SeluSeludense_23/BiasAdd:output:0*
T0*(
_output_shapes
:         ╝2
dense_23/Seluє
dropout_19/IdentityIdentitydense_23/Selu:activations:0*
T0*(
_output_shapes
:         ╝2
dropout_19/IdentityЕ
dense_24/MatMul/ReadVariableOpReadVariableOp'dense_24_matmul_readvariableop_resource*
_output_shapes
:	╝*
dtype02 
dense_24/MatMul/ReadVariableOpц
dense_24/MatMulMatMuldropout_19/Identity:output:0&dense_24/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/MatMulД
dense_24/BiasAdd/ReadVariableOpReadVariableOp(dense_24_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
dense_24/BiasAdd/ReadVariableOpЦ
dense_24/BiasAddBiasAdddense_24/MatMul:product:0'dense_24/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_24/BiasAdd|
dense_24/SigmoidSigmoiddense_24/BiasAdd:output:0*
T0*'
_output_shapes
:         2
dense_24/Sigmoidh
IdentityIdentitydense_24/Sigmoid:y:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Н
_input_shapes├
└:         :         :         :         :         :         :         :         :::::::::::S O
'
_output_shapes
:         
$
_user_specified_name
inputs/Age:XT
'
_output_shapes
:         
)
_user_specified_nameinputs/Embarked:TP
'
_output_shapes
:         
%
_user_specified_nameinputs/Fare:WS
'
_output_shapes
:         
(
_user_specified_nameinputs/IsAlone:ZV
'
_output_shapes
:         
+
_user_specified_nameinputs/NameLength:VR
'
_output_shapes
:         
'
_user_specified_nameinputs/Pclass:SO
'
_output_shapes
:         
$
_user_specified_name
inputs/Sex:UQ
'
_output_shapes
:         
&
_user_specified_nameinputs/Title:
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
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: "»L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*┬
serving_default«
3
Age,
serving_default_Age:0         
=
Embarked1
serving_default_Embarked:0         
5
Fare-
serving_default_Fare:0         
;
IsAlone0
serving_default_IsAlone:0	         
A

NameLength3
serving_default_NameLength:0         
9
Pclass/
serving_default_Pclass:0         
3
Sex,
serving_default_Sex:0	         
7
Title.
serving_default_Title:0         <
output_10
StatefulPartitionedCall:0         tensorflow/serving/predict:сш
ј]
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer-8

layer_with_weights-4

layer-9
	optimizer
_build_input_shape
	variables
trainable_variables
regularization_losses
	keras_api

signatures
А_default_save_signature
б__call__
+Б&call_and_return_all_conditional_losses"цY
_tf_keras_sequentialЁY{"class_name": "Sequential", "name": "sequential_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "sequential_4", "layers": [{"class_name": "DenseFeatures", "config": {"name": "dense_features_4", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "Age", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Embarked", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Fare", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "IsAlone", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "NameLength", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Pclass", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Sex", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Title", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 158, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 168, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 178, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 188, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"Pclass": {"class_name": "__tuple__", "items": [null, 1]}, "Sex": {"class_name": "__tuple__", "items": [null, 1]}, "Age": {"class_name": "__tuple__", "items": [null, 1]}, "Fare": {"class_name": "__tuple__", "items": [null, 1]}, "Embarked": {"class_name": "__tuple__", "items": [null, 1]}, "IsAlone": {"class_name": "__tuple__", "items": [null, 1]}, "NameLength": {"class_name": "__tuple__", "items": [null, 1]}, "Title": {"class_name": "__tuple__", "items": [null, 1]}}}, "build_input_shape": {"Pclass": {"class_name": "__tuple__", "items": [null, 1]}, "Sex": {"class_name": "__tuple__", "items": [null, 1]}, "Age": {"class_name": "__tuple__", "items": [null, 1]}, "Fare": {"class_name": "__tuple__", "items": [null, 1]}, "Embarked": {"class_name": "__tuple__", "items": [null, 1]}, "IsAlone": {"class_name": "__tuple__", "items": [null, 1]}, "NameLength": {"class_name": "__tuple__", "items": [null, 1]}, "Title": {"class_name": "__tuple__", "items": [null, 1]}}, "is_graph_network": false, "keras_version": "2.3.0-tf", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_4", "layers": [{"class_name": "DenseFeatures", "config": {"name": "dense_features_4", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "Age", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Embarked", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Fare", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "IsAlone", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "NameLength", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Pclass", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Sex", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Title", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 158, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 168, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 178, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 188, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}, {"class_name": "Dense", "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}], "build_input_shape": {"Pclass": {"class_name": "__tuple__", "items": [null, 1]}, "Sex": {"class_name": "__tuple__", "items": [null, 1]}, "Age": {"class_name": "__tuple__", "items": [null, 1]}, "Fare": {"class_name": "__tuple__", "items": [null, 1]}, "Embarked": {"class_name": "__tuple__", "items": [null, 1]}, "IsAlone": {"class_name": "__tuple__", "items": [null, 1]}, "NameLength": {"class_name": "__tuple__", "items": [null, 1]}, "Title": {"class_name": "__tuple__", "items": [null, 1]}}}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": true, "label_smoothing": 0}}, "metrics": ["accuracy"], "weighted_metrics": null, "loss_weights": null, "sample_weight_mode": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
»
_feature_columns

_resources
	variables
trainable_variables
regularization_losses
	keras_api
ц__call__
+Ц&call_and_return_all_conditional_losses"Э
_tf_keras_layerя{"class_name": "DenseFeatures", "name": "dense_features_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_features_4", "trainable": true, "dtype": "float32", "feature_columns": [{"class_name": "NumericColumn", "config": {"key": "Age", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Embarked", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Fare", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "IsAlone", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "NameLength", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Pclass", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Sex", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}, {"class_name": "NumericColumn", "config": {"key": "Title", "shape": {"class_name": "__tuple__", "items": [1]}, "default_value": null, "dtype": "float32", "normalizer_fn": null}}], "partitioner": null}, "build_input_shape": {"Pclass": {"class_name": "TensorShape", "items": [null, 1]}, "Sex": {"class_name": "TensorShape", "items": [null, 1]}, "Age": {"class_name": "TensorShape", "items": [null, 1]}, "Fare": {"class_name": "TensorShape", "items": [null, 1]}, "Embarked": {"class_name": "TensorShape", "items": [null, 1]}, "IsAlone": {"class_name": "TensorShape", "items": [null, 1]}, "NameLength": {"class_name": "TensorShape", "items": [null, 1]}, "Title": {"class_name": "TensorShape", "items": [null, 1]}}, "_is_feature_layer": true}
ќ

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
д__call__
+Д&call_and_return_all_conditional_losses"№
_tf_keras_layerН{"class_name": "Dense", "name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 158, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}
к
	variables
trainable_variables
 regularization_losses
!	keras_api
е__call__
+Е&call_and_return_all_conditional_losses"х
_tf_keras_layerЏ{"class_name": "Dropout", "name": "dropout_16", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_16", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
н

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
ф__call__
+Ф&call_and_return_all_conditional_losses"Г
_tf_keras_layerЊ{"class_name": "Dense", "name": "dense_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_21", "trainable": true, "dtype": "float32", "units": 168, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 158}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 158]}}
к
(	variables
)trainable_variables
*regularization_losses
+	keras_api
г__call__
+Г&call_and_return_all_conditional_losses"х
_tf_keras_layerЏ{"class_name": "Dropout", "name": "dropout_17", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_17", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
н

,kernel
-bias
.	variables
/trainable_variables
0regularization_losses
1	keras_api
«__call__
+»&call_and_return_all_conditional_losses"Г
_tf_keras_layerЊ{"class_name": "Dense", "name": "dense_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_22", "trainable": true, "dtype": "float32", "units": 178, "activation": "swish", "use_bias": true, "kernel_initializer": {"class_name": "GlorotNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 168}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 168]}}
к
2	variables
3trainable_variables
4regularization_losses
5	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"х
_tf_keras_layerЏ{"class_name": "Dropout", "name": "dropout_18", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_18", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
џ

6kernel
7bias
8	variables
9trainable_variables
:regularization_losses
;	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"з
_tf_keras_layer┘{"class_name": "Dense", "name": "dense_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_23", "trainable": true, "dtype": "float32", "units": 188, "activation": "selu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 178}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 178]}}
к
<	variables
=trainable_variables
>regularization_losses
?	keras_api
┤__call__
+х&call_and_return_all_conditional_losses"х
_tf_keras_layerЏ{"class_name": "Dropout", "name": "dropout_19", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dropout_19", "trainable": true, "dtype": "float32", "rate": 0.2, "noise_shape": null, "seed": null}}
Џ

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
Х__call__
+и&call_and_return_all_conditional_losses"З
_tf_keras_layer┌{"class_name": "Dense", "name": "dense_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "config": {"name": "dense_24", "trainable": true, "dtype": "float32", "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 1.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 188}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 188]}}
Џ
Fiter

Gbeta_1

Hbeta_2
	Idecay
Jlearning_ratemЇmј"mЈ#mљ,mЉ-mњ6mЊ7mћ@mЋAmќvЌvў"vЎ#vџ,vЏ-vю6vЮ7vъ@vЪAvа"
	optimizer
 "
trackable_dict_wrapper
f
0
1
"2
#3
,4
-5
66
77
@8
A9"
trackable_list_wrapper
f
0
1
"2
#3
,4
-5
66
77
@8
A9"
trackable_list_wrapper
 "
trackable_list_wrapper
╬
	variables
trainable_variables
Klayer_metrics
Llayer_regularization_losses
Mmetrics

Nlayers
regularization_losses
Onon_trainable_variables
б__call__
А_default_save_signature
+Б&call_and_return_all_conditional_losses
'Б"call_and_return_conditional_losses"
_generic_user_object
-
Иserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
trainable_variables
Player_metrics
Qlayer_regularization_losses
Rmetrics

Slayers
regularization_losses
Tnon_trainable_variables
ц__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
/:-	ъ2sequential_4/dense_20/kernel
):'ъ2sequential_4/dense_20/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
trainable_variables
Ulayer_metrics
Vlayer_regularization_losses
Wmetrics

Xlayers
regularization_losses
Ynon_trainable_variables
д__call__
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
	variables
trainable_variables
Zlayer_metrics
[layer_regularization_losses
\metrics

]layers
 regularization_losses
^non_trainable_variables
е__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
0:.
ъе2sequential_4/dense_21/kernel
):'е2sequential_4/dense_21/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
$	variables
%trainable_variables
_layer_metrics
`layer_regularization_losses
ametrics

blayers
&regularization_losses
cnon_trainable_variables
ф__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
(	variables
)trainable_variables
dlayer_metrics
elayer_regularization_losses
fmetrics

glayers
*regularization_losses
hnon_trainable_variables
г__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
0:.
е▓2sequential_4/dense_22/kernel
):'▓2sequential_4/dense_22/bias
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
░
.	variables
/trainable_variables
ilayer_metrics
jlayer_regularization_losses
kmetrics

llayers
0regularization_losses
mnon_trainable_variables
«__call__
+»&call_and_return_all_conditional_losses
'»"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
2	variables
3trainable_variables
nlayer_metrics
olayer_regularization_losses
pmetrics

qlayers
4regularization_losses
rnon_trainable_variables
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
0:.
▓╝2sequential_4/dense_23/kernel
):'╝2sequential_4/dense_23/bias
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
 "
trackable_list_wrapper
░
8	variables
9trainable_variables
slayer_metrics
tlayer_regularization_losses
umetrics

vlayers
:regularization_losses
wnon_trainable_variables
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
<	variables
=trainable_variables
xlayer_metrics
ylayer_regularization_losses
zmetrics

{layers
>regularization_losses
|non_trainable_variables
┤__call__
+х&call_and_return_all_conditional_losses
'х"call_and_return_conditional_losses"
_generic_user_object
/:-	╝2sequential_4/dense_24/kernel
(:&2sequential_4/dense_24/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
B	variables
Ctrainable_variables
}layer_metrics
~layer_regularization_losses
metrics
ђlayers
Dregularization_losses
Ђnon_trainable_variables
Х__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
0
ѓ0
Ѓ1"
trackable_list_wrapper
f
0
1
2
3
4
5
6
7
	8

9"
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
┐

ёtotal

Ёcount
є	variables
Є	keras_api"ё
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 

ѕtotal

Ѕcount
і
_fn_kwargs
І	variables
ї	keras_api"│
_tf_keras_metricў{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "binary_accuracy"}}
:  (2total
:  (2count
0
ё0
Ё1"
trackable_list_wrapper
.
є	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
ѕ0
Ѕ1"
trackable_list_wrapper
.
І	variables"
_generic_user_object
4:2	ъ2#Adam/sequential_4/dense_20/kernel/m
.:,ъ2!Adam/sequential_4/dense_20/bias/m
5:3
ъе2#Adam/sequential_4/dense_21/kernel/m
.:,е2!Adam/sequential_4/dense_21/bias/m
5:3
е▓2#Adam/sequential_4/dense_22/kernel/m
.:,▓2!Adam/sequential_4/dense_22/bias/m
5:3
▓╝2#Adam/sequential_4/dense_23/kernel/m
.:,╝2!Adam/sequential_4/dense_23/bias/m
4:2	╝2#Adam/sequential_4/dense_24/kernel/m
-:+2!Adam/sequential_4/dense_24/bias/m
4:2	ъ2#Adam/sequential_4/dense_20/kernel/v
.:,ъ2!Adam/sequential_4/dense_20/bias/v
5:3
ъе2#Adam/sequential_4/dense_21/kernel/v
.:,е2!Adam/sequential_4/dense_21/bias/v
5:3
е▓2#Adam/sequential_4/dense_22/kernel/v
.:,▓2!Adam/sequential_4/dense_22/bias/v
5:3
▓╝2#Adam/sequential_4/dense_23/kernel/v
.:,╝2!Adam/sequential_4/dense_23/bias/v
4:2	╝2#Adam/sequential_4/dense_24/kernel/v
-:+2!Adam/sequential_4/dense_24/bias/v
А2ъ
!__inference__wrapped_model_212496Э
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *убс
Яф▄
$
Ageі
Age         
.
Embarked"і
Embarked         
&
Fareі
Fare         
,
IsAlone!і
IsAlone         	
2

NameLength$і!

NameLength         
*
Pclass і
Pclass         
$
Sexі
Sex         	
(
Titleі
Title         
ѓ2 
-__inference_sequential_4_layer_call_fn_213000
-__inference_sequential_4_layer_call_fn_213079
-__inference_sequential_4_layer_call_fn_213449
-__inference_sequential_4_layer_call_fn_213481└
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
Ь2в
H__inference_sequential_4_layer_call_and_return_conditional_losses_213417
H__inference_sequential_4_layer_call_and_return_conditional_losses_212873
H__inference_sequential_4_layer_call_and_return_conditional_losses_213283
H__inference_sequential_4_layer_call_and_return_conditional_losses_212920└
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
Ч2щ
1__inference_dense_features_4_layer_call_fn_213572├
║▓Х
FullArgSpec9
args1џ.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ќ2ћ
L__inference_dense_features_4_layer_call_and_return_conditional_losses_213560├
║▓Х
FullArgSpec9
args1џ.
jself

jfeatures
jcols_to_output_tensors
varargs
 
varkw
 
defaultsб

 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
М2л
)__inference_dense_20_layer_call_fn_213592б
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
D__inference_dense_20_layer_call_and_return_conditional_losses_213583б
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
ћ2Љ
+__inference_dropout_16_layer_call_fn_213619
+__inference_dropout_16_layer_call_fn_213614┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
╩2К
F__inference_dropout_16_layer_call_and_return_conditional_losses_213609
F__inference_dropout_16_layer_call_and_return_conditional_losses_213604┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
М2л
)__inference_dense_21_layer_call_fn_213644б
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
D__inference_dense_21_layer_call_and_return_conditional_losses_213635б
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
ћ2Љ
+__inference_dropout_17_layer_call_fn_213666
+__inference_dropout_17_layer_call_fn_213671┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
╩2К
F__inference_dropout_17_layer_call_and_return_conditional_losses_213656
F__inference_dropout_17_layer_call_and_return_conditional_losses_213661┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
М2л
)__inference_dense_22_layer_call_fn_213696б
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
D__inference_dense_22_layer_call_and_return_conditional_losses_213687б
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
ћ2Љ
+__inference_dropout_18_layer_call_fn_213723
+__inference_dropout_18_layer_call_fn_213718┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
╩2К
F__inference_dropout_18_layer_call_and_return_conditional_losses_213708
F__inference_dropout_18_layer_call_and_return_conditional_losses_213713┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
М2л
)__inference_dense_23_layer_call_fn_213743б
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
D__inference_dense_23_layer_call_and_return_conditional_losses_213734б
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
ћ2Љ
+__inference_dropout_19_layer_call_fn_213770
+__inference_dropout_19_layer_call_fn_213765┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
╩2К
F__inference_dropout_19_layer_call_and_return_conditional_losses_213760
F__inference_dropout_19_layer_call_and_return_conditional_losses_213755┤
Ф▓Д
FullArgSpec)
args!џ
jself
jinputs

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
М2л
)__inference_dense_24_layer_call_fn_213790б
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
D__inference_dense_24_layer_call_and_return_conditional_losses_213781б
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
fBd
$__inference_signature_wrapper_213121AgeEmbarkedFareIsAlone
NameLengthPclassSexTitleП
!__inference__wrapped_model_212496и
"#,-67@Aзб№
убс
Яф▄
$
Ageі
Age         
.
Embarked"і
Embarked         
&
Fareі
Fare         
,
IsAlone!і
IsAlone         	
2

NameLength$і!

NameLength         
*
Pclass і
Pclass         
$
Sexі
Sex         	
(
Titleі
Title         
ф "3ф0
.
output_1"і
output_1         Ц
D__inference_dense_20_layer_call_and_return_conditional_losses_213583]/б,
%б"
 і
inputs         
ф "&б#
і
0         ъ
џ }
)__inference_dense_20_layer_call_fn_213592P/б,
%б"
 і
inputs         
ф "і         ъд
D__inference_dense_21_layer_call_and_return_conditional_losses_213635^"#0б-
&б#
!і
inputs         ъ
ф "&б#
і
0         е
џ ~
)__inference_dense_21_layer_call_fn_213644Q"#0б-
&б#
!і
inputs         ъ
ф "і         ед
D__inference_dense_22_layer_call_and_return_conditional_losses_213687^,-0б-
&б#
!і
inputs         е
ф "&б#
і
0         ▓
џ ~
)__inference_dense_22_layer_call_fn_213696Q,-0б-
&б#
!і
inputs         е
ф "і         ▓д
D__inference_dense_23_layer_call_and_return_conditional_losses_213734^670б-
&б#
!і
inputs         ▓
ф "&б#
і
0         ╝
џ ~
)__inference_dense_23_layer_call_fn_213743Q670б-
&б#
!і
inputs         ▓
ф "і         ╝Ц
D__inference_dense_24_layer_call_and_return_conditional_losses_213781]@A0б-
&б#
!і
inputs         ╝
ф "%б"
і
0         
џ }
)__inference_dense_24_layer_call_fn_213790P@A0б-
&б#
!і
inputs         ╝
ф "і         ║
L__inference_dense_features_4_layer_call_and_return_conditional_losses_213560ж┐б╗
│б»
ефц
-
Age&і#
features/Age         
7
Embarked+і(
features/Embarked         
/
Fare'і$
features/Fare         
5
IsAlone*і'
features/IsAlone         	
;

NameLength-і*
features/NameLength         
3
Pclass)і&
features/Pclass         
-
Sex&і#
features/Sex         	
1
Title(і%
features/Title         

 
ф "%б"
і
0         
џ њ
1__inference_dense_features_4_layer_call_fn_213572▄┐б╗
│б»
ефц
-
Age&і#
features/Age         
7
Embarked+і(
features/Embarked         
/
Fare'і$
features/Fare         
5
IsAlone*і'
features/IsAlone         	
;

NameLength-і*
features/NameLength         
3
Pclass)і&
features/Pclass         
-
Sex&і#
features/Sex         	
1
Title(і%
features/Title         

 
ф "і         е
F__inference_dropout_16_layer_call_and_return_conditional_losses_213604^4б1
*б'
!і
inputs         ъ
p
ф "&б#
і
0         ъ
џ е
F__inference_dropout_16_layer_call_and_return_conditional_losses_213609^4б1
*б'
!і
inputs         ъ
p 
ф "&б#
і
0         ъ
џ ђ
+__inference_dropout_16_layer_call_fn_213614Q4б1
*б'
!і
inputs         ъ
p
ф "і         ъђ
+__inference_dropout_16_layer_call_fn_213619Q4б1
*б'
!і
inputs         ъ
p 
ф "і         ъе
F__inference_dropout_17_layer_call_and_return_conditional_losses_213656^4б1
*б'
!і
inputs         е
p
ф "&б#
і
0         е
џ е
F__inference_dropout_17_layer_call_and_return_conditional_losses_213661^4б1
*б'
!і
inputs         е
p 
ф "&б#
і
0         е
џ ђ
+__inference_dropout_17_layer_call_fn_213666Q4б1
*б'
!і
inputs         е
p
ф "і         еђ
+__inference_dropout_17_layer_call_fn_213671Q4б1
*б'
!і
inputs         е
p 
ф "і         ее
F__inference_dropout_18_layer_call_and_return_conditional_losses_213708^4б1
*б'
!і
inputs         ▓
p
ф "&б#
і
0         ▓
џ е
F__inference_dropout_18_layer_call_and_return_conditional_losses_213713^4б1
*б'
!і
inputs         ▓
p 
ф "&б#
і
0         ▓
џ ђ
+__inference_dropout_18_layer_call_fn_213718Q4б1
*б'
!і
inputs         ▓
p
ф "і         ▓ђ
+__inference_dropout_18_layer_call_fn_213723Q4б1
*б'
!і
inputs         ▓
p 
ф "і         ▓е
F__inference_dropout_19_layer_call_and_return_conditional_losses_213755^4б1
*б'
!і
inputs         ╝
p
ф "&б#
і
0         ╝
џ е
F__inference_dropout_19_layer_call_and_return_conditional_losses_213760^4б1
*б'
!і
inputs         ╝
p 
ф "&б#
і
0         ╝
џ ђ
+__inference_dropout_19_layer_call_fn_213765Q4б1
*б'
!і
inputs         ╝
p
ф "і         ╝ђ
+__inference_dropout_19_layer_call_fn_213770Q4б1
*б'
!і
inputs         ╝
p 
ф "і         ╝■
H__inference_sequential_4_layer_call_and_return_conditional_losses_212873▒
"#,-67@Aчбэ
№бв
Яф▄
$
Ageі
Age         
.
Embarked"і
Embarked         
&
Fareі
Fare         
,
IsAlone!і
IsAlone         	
2

NameLength$і!

NameLength         
*
Pclass і
Pclass         
$
Sexі
Sex         	
(
Titleі
Title         
p

 
ф "%б"
і
0         
џ ■
H__inference_sequential_4_layer_call_and_return_conditional_losses_212920▒
"#,-67@Aчбэ
№бв
Яф▄
$
Ageі
Age         
.
Embarked"і
Embarked         
&
Fareі
Fare         
,
IsAlone!і
IsAlone         	
2

NameLength$і!

NameLength         
*
Pclass і
Pclass         
$
Sexі
Sex         	
(
Titleі
Title         
p 

 
ф "%б"
і
0         
џ Х
H__inference_sequential_4_layer_call_and_return_conditional_losses_213283ж
"#,-67@A│б»
ДбБ
ўфћ
+
Age$і!

inputs/Age         
5
Embarked)і&
inputs/Embarked         
-
Fare%і"
inputs/Fare         
3
IsAlone(і%
inputs/IsAlone         	
9

NameLength+і(
inputs/NameLength         
1
Pclass'і$
inputs/Pclass         
+
Sex$і!

inputs/Sex         	
/
Title&і#
inputs/Title         
p

 
ф "%б"
і
0         
џ Х
H__inference_sequential_4_layer_call_and_return_conditional_losses_213417ж
"#,-67@A│б»
ДбБ
ўфћ
+
Age$і!

inputs/Age         
5
Embarked)і&
inputs/Embarked         
-
Fare%і"
inputs/Fare         
3
IsAlone(і%
inputs/IsAlone         	
9

NameLength+і(
inputs/NameLength         
1
Pclass'і$
inputs/Pclass         
+
Sex$і!

inputs/Sex         	
/
Title&і#
inputs/Title         
p 

 
ф "%б"
і
0         
џ о
-__inference_sequential_4_layer_call_fn_213000ц
"#,-67@Aчбэ
№бв
Яф▄
$
Ageі
Age         
.
Embarked"і
Embarked         
&
Fareі
Fare         
,
IsAlone!і
IsAlone         	
2

NameLength$і!

NameLength         
*
Pclass і
Pclass         
$
Sexі
Sex         	
(
Titleі
Title         
p

 
ф "і         о
-__inference_sequential_4_layer_call_fn_213079ц
"#,-67@Aчбэ
№бв
Яф▄
$
Ageі
Age         
.
Embarked"і
Embarked         
&
Fareі
Fare         
,
IsAlone!і
IsAlone         	
2

NameLength$і!

NameLength         
*
Pclass і
Pclass         
$
Sexі
Sex         	
(
Titleі
Title         
p 

 
ф "і         ј
-__inference_sequential_4_layer_call_fn_213449▄
"#,-67@A│б»
ДбБ
ўфћ
+
Age$і!

inputs/Age         
5
Embarked)і&
inputs/Embarked         
-
Fare%і"
inputs/Fare         
3
IsAlone(і%
inputs/IsAlone         	
9

NameLength+і(
inputs/NameLength         
1
Pclass'і$
inputs/Pclass         
+
Sex$і!

inputs/Sex         	
/
Title&і#
inputs/Title         
p

 
ф "і         ј
-__inference_sequential_4_layer_call_fn_213481▄
"#,-67@A│б»
ДбБ
ўфћ
+
Age$і!

inputs/Age         
5
Embarked)і&
inputs/Embarked         
-
Fare%і"
inputs/Fare         
3
IsAlone(і%
inputs/IsAlone         	
9

NameLength+і(
inputs/NameLength         
1
Pclass'і$
inputs/Pclass         
+
Sex$і!

inputs/Sex         	
/
Title&і#
inputs/Title         
p 

 
ф "і         ┘
$__inference_signature_wrapper_213121░
"#,-67@AВбУ
б 
Яф▄
$
Ageі
Age         
.
Embarked"і
Embarked         
&
Fareі
Fare         
,
IsAlone!і
IsAlone         	
2

NameLength$і!

NameLength         
*
Pclass і
Pclass         
$
Sexі
Sex         	
(
Titleі
Title         "3ф0
.
output_1"і
output_1         