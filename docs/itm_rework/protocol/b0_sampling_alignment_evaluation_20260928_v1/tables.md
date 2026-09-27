# 固定R guidance完整数值表

旧R=2.5/1/1，S1=1/1/1。每格3来源等权；contact仅val000433/000382，n=2，train和val000139为N/A。所有来源min/max与逐条轴向extent/root path在JSON中保留。误差差值为前项减后项；jerk是绝对量，须同GT解释。

## 全部40格

| arm | split | config | condition | global m | relative m | root m | joint proxy m | velocity m/s | jerk | GT jerk | contact m/s | GT contact |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R_static_reference_delta | train | head | paired | 0.620450 | 0.218094 | 0.603858 | 0.624087 | 0.743013 | 65.463642 | 50.256844 | N/A | N/A |
| R_static_reference_delta | train | head | physical_static | 0.754257 | 0.282879 | 0.694424 | 0.769927 | 1.122086 | 101.545697 | 50.256844 | N/A | N/A |
| R_static_reference_delta | train | head | text_only | 0.754257 | 0.282879 | 0.694424 | 0.769927 | 1.122086 | 101.545697 | 50.256844 | N/A | N/A |
| R_static_reference_delta | train | wrists | paired | 0.404349 | 0.143198 | 0.420371 | 0.426309 | 0.549521 | 60.750056 | 50.256844 | N/A | N/A |
| R_static_reference_delta | train | wrists | physical_static | 0.834034 | 0.302299 | 0.741714 | 0.958895 | 1.127934 | 83.752317 | 50.256844 | N/A | N/A |
| R_static_reference_delta | train | wrists | text_only | 0.834034 | 0.302299 | 0.741714 | 0.958895 | 1.127934 | 83.752317 | 50.256844 | N/A | N/A |
| R_static_reference_delta | val | head | backbone_bypass | 0.754609 | 0.128406 | 0.770747 | 0.716031 | 0.875300 | 45.319133 | 30.486643 | 0.136560 | 0.045625 |
| R_static_reference_delta | val | head | imu_only | 0.670188 | 0.165399 | 0.699632 | 0.659390 | 1.060066 | 62.622525 | 30.486643 | 0.378744 | 0.045625 |
| R_static_reference_delta | val | head | numeric_zero | 0.591301 | 0.139290 | 0.573391 | 0.580573 | 0.803002 | 38.113588 | 30.486643 | 0.148208 | 0.045625 |
| R_static_reference_delta | val | head | paired | 0.729000 | 0.124456 | 0.742919 | 0.690830 | 0.863915 | 47.248325 | 30.486643 | 0.138387 | 0.045625 |
| R_static_reference_delta | val | head | physical_static | 0.754609 | 0.128406 | 0.770747 | 0.716031 | 0.875300 | 45.319133 | 30.486643 | 0.136560 | 0.045625 |
| R_static_reference_delta | val | head | tail_reversed | 0.763025 | 0.120627 | 0.779694 | 0.732004 | 0.849254 | 44.502775 | 30.486643 | 0.131566 | 0.045625 |
| R_static_reference_delta | val | head | text_only | 0.754609 | 0.128406 | 0.770747 | 0.716031 | 0.875300 | 45.319133 | 30.486643 | 0.136560 | 0.045625 |
| R_static_reference_delta | val | wrists | backbone_bypass | 0.807173 | 0.126388 | 0.808650 | 0.938831 | 0.718608 | 22.584989 | 30.486643 | 0.040871 | 0.045625 |
| R_static_reference_delta | val | wrists | imu_only | 2.282761 | 0.256165 | 2.393814 | 2.345387 | 1.887663 | 178.578895 | 30.486643 | 1.583199 | 0.045625 |
| R_static_reference_delta | val | wrists | numeric_zero | 1.159760 | 0.237755 | 1.159129 | 1.317970 | 0.799625 | 33.049562 | 30.486643 | 0.082569 | 0.045625 |
| R_static_reference_delta | val | wrists | paired | 0.914202 | 0.196429 | 0.865027 | 0.969227 | 0.717674 | 33.516244 | 30.486643 | 0.125433 | 0.045625 |
| R_static_reference_delta | val | wrists | physical_static | 0.807173 | 0.126388 | 0.808650 | 0.938831 | 0.718608 | 22.584989 | 30.486643 | 0.040871 | 0.045625 |
| R_static_reference_delta | val | wrists | tail_reversed | 0.854535 | 0.194673 | 0.802864 | 0.912119 | 0.750041 | 33.918675 | 30.486643 | 0.130407 | 0.045625 |
| R_static_reference_delta | val | wrists | text_only | 0.807173 | 0.126388 | 0.808650 | 0.938831 | 0.718608 | 22.584989 | 30.486643 | 0.040871 | 0.045625 |
| S1_R_unit_guidance | train | head | paired | 0.856504 | 0.193923 | 0.934674 | 0.808005 | 0.864609 | 63.068075 | 50.256844 | N/A | N/A |
| S1_R_unit_guidance | train | head | physical_static | 1.156793 | 0.287798 | 1.199184 | 1.101772 | 1.125387 | 44.096455 | 50.256844 | N/A | N/A |
| S1_R_unit_guidance | train | head | text_only | 1.156793 | 0.287798 | 1.199184 | 1.101772 | 1.125387 | 44.096455 | 50.256844 | N/A | N/A |
| S1_R_unit_guidance | train | wrists | paired | 0.352544 | 0.124033 | 0.369669 | 0.334859 | 0.562212 | 56.571948 | 50.256844 | N/A | N/A |
| S1_R_unit_guidance | train | wrists | physical_static | 0.924578 | 0.281299 | 0.825906 | 1.038270 | 0.937986 | 51.135813 | 50.256844 | N/A | N/A |
| S1_R_unit_guidance | train | wrists | text_only | 0.924578 | 0.281299 | 0.825906 | 1.038270 | 0.937986 | 51.135813 | 50.256844 | N/A | N/A |
| S1_R_unit_guidance | val | head | backbone_bypass | 0.525084 | 0.143343 | 0.515623 | 0.538234 | 0.902402 | 42.806522 | 30.486643 | 0.144132 | 0.045625 |
| S1_R_unit_guidance | val | head | imu_only | 0.670188 | 0.165399 | 0.699632 | 0.659390 | 1.060066 | 62.622489 | 30.486643 | 0.378745 | 0.045625 |
| S1_R_unit_guidance | val | head | numeric_zero | 0.635482 | 0.160413 | 0.670047 | 0.680257 | 0.884169 | 42.847374 | 30.486643 | 0.179257 | 0.045625 |
| S1_R_unit_guidance | val | head | paired | 0.445638 | 0.132352 | 0.411567 | 0.449433 | 0.884035 | 44.212800 | 30.486643 | 0.154043 | 0.045625 |
| S1_R_unit_guidance | val | head | physical_static | 0.525084 | 0.143343 | 0.515623 | 0.538234 | 0.902402 | 42.806522 | 30.486643 | 0.144132 | 0.045625 |
| S1_R_unit_guidance | val | head | tail_reversed | 0.430870 | 0.124746 | 0.398150 | 0.437676 | 0.844634 | 38.831537 | 30.486643 | 0.134074 | 0.045625 |
| S1_R_unit_guidance | val | head | text_only | 0.525084 | 0.143343 | 0.515623 | 0.538234 | 0.902402 | 42.806522 | 30.486643 | 0.144132 | 0.045625 |
| S1_R_unit_guidance | val | wrists | backbone_bypass | 0.812157 | 0.133567 | 0.845118 | 0.953546 | 0.802322 | 31.606018 | 30.486643 | 0.133958 | 0.045625 |
| S1_R_unit_guidance | val | wrists | imu_only | 2.282761 | 0.256165 | 2.393814 | 2.345387 | 1.887663 | 178.578747 | 30.486643 | 1.583199 | 0.045625 |
| S1_R_unit_guidance | val | wrists | numeric_zero | 1.172175 | 0.262768 | 1.120979 | 1.343887 | 0.894580 | 49.871400 | 30.486643 | 0.121611 | 0.045625 |
| S1_R_unit_guidance | val | wrists | paired | 0.819037 | 0.203696 | 0.771879 | 0.887520 | 0.737537 | 35.603048 | 30.486643 | 0.199036 | 0.045625 |
| S1_R_unit_guidance | val | wrists | physical_static | 0.812157 | 0.133567 | 0.845118 | 0.953546 | 0.802322 | 31.606018 | 30.486643 | 0.133958 | 0.045625 |
| S1_R_unit_guidance | val | wrists | tail_reversed | 0.821394 | 0.201981 | 0.791574 | 0.887360 | 0.812900 | 39.801512 | 30.486643 | 0.212710 | 0.045625 |
| S1_R_unit_guidance | val | wrists | text_only | 0.812157 | 0.133567 | 0.845118 | 0.953546 | 0.802322 | 31.606018 | 30.486643 | 0.133958 | 0.045625 |

## 全部S1−旧R匹配条件

| arm | split | config | comparison | Δglobal | Δrelative | Δroot | Δproxy | Δvelocity |
|---|---|---|---|---:|---:|---:|---:|---:|
| S1−oldR | train | head | paired | 0.236054 | -0.024171 | 0.330816 | 0.183919 | 0.121596 |
| S1−oldR | train | head | physical_static | 0.402536 | 0.004919 | 0.504760 | 0.331844 | 0.003301 |
| S1−oldR | train | head | text_only | 0.402536 | 0.004919 | 0.504760 | 0.331844 | 0.003301 |
| S1−oldR | train | wrists | paired | -0.051805 | -0.019165 | -0.050703 | -0.091450 | 0.012691 |
| S1−oldR | train | wrists | physical_static | 0.090544 | -0.021000 | 0.084191 | 0.079375 | -0.189947 |
| S1−oldR | train | wrists | text_only | 0.090544 | -0.021000 | 0.084191 | 0.079375 | -0.189947 |
| S1−oldR | val | head | backbone_bypass | -0.229525 | 0.014937 | -0.255124 | -0.177797 | 0.027102 |
| S1−oldR | val | head | imu_only | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 |
| S1−oldR | val | head | numeric_zero | 0.044182 | 0.021123 | 0.096656 | 0.099685 | 0.081167 |
| S1−oldR | val | head | paired | -0.283362 | 0.007896 | -0.331353 | -0.241397 | 0.020120 |
| S1−oldR | val | head | physical_static | -0.229525 | 0.014937 | -0.255124 | -0.177797 | 0.027102 |
| S1−oldR | val | head | tail_reversed | -0.332156 | 0.004119 | -0.381544 | -0.294328 | -0.004620 |
| S1−oldR | val | head | text_only | -0.229525 | 0.014937 | -0.255124 | -0.177797 | 0.027102 |
| S1−oldR | val | wrists | backbone_bypass | 0.004984 | 0.007179 | 0.036468 | 0.014715 | 0.083714 |
| S1−oldR | val | wrists | imu_only | -0.000000 | -0.000000 | -0.000000 | -0.000000 | -0.000000 |
| S1−oldR | val | wrists | numeric_zero | 0.012415 | 0.025013 | -0.038150 | 0.025917 | 0.094955 |
| S1−oldR | val | wrists | paired | -0.095165 | 0.007267 | -0.093149 | -0.081707 | 0.019863 |
| S1−oldR | val | wrists | physical_static | 0.004984 | 0.007179 | 0.036468 | 0.014715 | 0.083714 |
| S1−oldR | val | wrists | tail_reversed | -0.033140 | 0.007308 | -0.011290 | -0.024759 | 0.062860 |
| S1−oldR | val | wrists | text_only | 0.004984 | 0.007179 | 0.036468 | 0.014715 | 0.083714 |

## 各guidance内paired−comparator

| arm | split | config | comparison | Δglobal | Δrelative | Δroot | Δproxy | Δvelocity |
|---|---|---|---|---:|---:|---:|---:|---:|
| R_static_reference_delta | train | head | physical_static | -0.133807 | -0.064785 | -0.090566 | -0.145841 | -0.379073 |
| R_static_reference_delta | train | head | text_only | -0.133807 | -0.064785 | -0.090566 | -0.145841 | -0.379073 |
| R_static_reference_delta | train | wrists | physical_static | -0.429685 | -0.159101 | -0.321343 | -0.532586 | -0.578413 |
| R_static_reference_delta | train | wrists | text_only | -0.429685 | -0.159101 | -0.321343 | -0.532586 | -0.578413 |
| R_static_reference_delta | val | head | numeric_zero | 0.137699 | -0.014834 | 0.169528 | 0.110257 | 0.060913 |
| R_static_reference_delta | val | head | physical_static | -0.025609 | -0.003950 | -0.027827 | -0.025201 | -0.011385 |
| R_static_reference_delta | val | head | tail_reversed | -0.034026 | 0.003828 | -0.036775 | -0.041175 | 0.014661 |
| R_static_reference_delta | val | head | text_only | -0.025609 | -0.003950 | -0.027827 | -0.025201 | -0.011385 |
| R_static_reference_delta | val | wrists | numeric_zero | -0.245558 | -0.041326 | -0.294101 | -0.348743 | -0.081952 |
| R_static_reference_delta | val | wrists | physical_static | 0.107029 | 0.070041 | 0.056378 | 0.030396 | -0.000934 |
| R_static_reference_delta | val | wrists | tail_reversed | 0.059668 | 0.001756 | 0.062163 | 0.057108 | -0.032367 |
| R_static_reference_delta | val | wrists | text_only | 0.107029 | 0.070041 | 0.056378 | 0.030396 | -0.000934 |
| S1_R_unit_guidance | train | head | physical_static | -0.300289 | -0.093875 | -0.264510 | -0.293766 | -0.260778 |
| S1_R_unit_guidance | train | head | text_only | -0.300289 | -0.093875 | -0.264510 | -0.293766 | -0.260778 |
| S1_R_unit_guidance | train | wrists | physical_static | -0.572034 | -0.157266 | -0.456237 | -0.703412 | -0.375774 |
| S1_R_unit_guidance | train | wrists | text_only | -0.572034 | -0.157266 | -0.456237 | -0.703412 | -0.375774 |
| S1_R_unit_guidance | val | head | numeric_zero | -0.189845 | -0.028061 | -0.258480 | -0.230824 | -0.000134 |
| S1_R_unit_guidance | val | head | physical_static | -0.079447 | -0.010991 | -0.104056 | -0.088801 | -0.018367 |
| S1_R_unit_guidance | val | head | tail_reversed | 0.014768 | 0.007606 | 0.013416 | 0.011757 | 0.039401 |
| S1_R_unit_guidance | val | head | text_only | -0.079447 | -0.010991 | -0.104056 | -0.088801 | -0.018367 |
| S1_R_unit_guidance | val | wrists | numeric_zero | -0.353138 | -0.059072 | -0.349100 | -0.456367 | -0.157043 |
| S1_R_unit_guidance | val | wrists | physical_static | 0.006880 | 0.070129 | -0.073239 | -0.066026 | -0.064786 |
| S1_R_unit_guidance | val | wrists | tail_reversed | -0.002357 | 0.001715 | -0.019695 | 0.000160 | -0.075364 |
| S1_R_unit_guidance | val | wrists | text_only | 0.006880 | 0.070129 | -0.073239 | -0.066026 | -0.064786 |

## (S1 paired−comparator)−(旧R paired−comparator)

| arm | split | config | comparison | Δglobal | Δrelative | Δroot | Δproxy | Δvelocity |
|---|---|---|---|---:|---:|---:|---:|---:|
| S1−oldR | train | head | physical_static | -0.166482 | -0.029090 | -0.173944 | -0.147926 | 0.118295 |
| S1−oldR | train | head | text_only | -0.166482 | -0.029090 | -0.173944 | -0.147926 | 0.118295 |
| S1−oldR | train | wrists | physical_static | -0.142349 | 0.001835 | -0.134894 | -0.170825 | 0.202638 |
| S1−oldR | train | wrists | text_only | -0.142349 | 0.001835 | -0.134894 | -0.170825 | 0.202638 |
| S1−oldR | val | head | numeric_zero | -0.327544 | -0.013227 | -0.428009 | -0.341081 | -0.061047 |
| S1−oldR | val | head | physical_static | -0.053837 | -0.007041 | -0.076229 | -0.063600 | -0.006982 |
| S1−oldR | val | head | tail_reversed | 0.048794 | 0.003777 | 0.050191 | 0.052931 | 0.024740 |
| S1−oldR | val | head | text_only | -0.053837 | -0.007041 | -0.076229 | -0.063600 | -0.006982 |
| S1−oldR | val | wrists | numeric_zero | -0.107580 | -0.017747 | -0.054999 | -0.107624 | -0.075092 |
| S1−oldR | val | wrists | physical_static | -0.100149 | 0.000088 | -0.129617 | -0.096422 | -0.063851 |
| S1−oldR | val | wrists | tail_reversed | -0.062024 | -0.000041 | -0.081858 | -0.056948 | -0.042997 |
| S1−oldR | val | wrists | text_only | -0.100149 | 0.000088 | -0.129617 | -0.096422 | -0.063851 |

## Q：全部条件的原量与分解

同来源先平均2次noise replicate，随后3来源等权；按t分别报告。legacy为float64概念重组，Q无f01，不是四分支float32轨迹重放。

| split | config | t | condition | unit MSE | legacy MSE | legacy−unit | cross | energy |
|---|---|---:|---|---:|---:|---:|---:|---:|
| train | head | 50 | paired | 0.013242 | 0.013378 | 0.000136 | -0.000014 | 0.000150 |
| train | head | 50 | physical_static | 0.016947 | 0.017069 | 0.000122 | -0.000028 | 0.000150 |
| train | head | 50 | tail_reversed | 0.024005 | 0.024172 | 0.000167 | 0.000017 | 0.000150 |
| train | head | 250 | paired | 0.015401 | 0.015579 | 0.000177 | 0.000000 | 0.000177 |
| train | head | 250 | physical_static | 0.020150 | 0.020291 | 0.000141 | -0.000036 | 0.000177 |
| train | head | 250 | tail_reversed | 0.027002 | 0.027206 | 0.000204 | 0.000027 | 0.000177 |
| train | head | 500 | paired | 0.023362 | 0.023738 | 0.000376 | 0.000057 | 0.000318 |
| train | head | 500 | physical_static | 0.031258 | 0.031591 | 0.000333 | 0.000014 | 0.000318 |
| train | head | 500 | tail_reversed | 0.037541 | 0.037932 | 0.000391 | 0.000073 | 0.000318 |
| train | head | 900 | paired | 0.099181 | 0.105299 | 0.006118 | -0.000030 | 0.006148 |
| train | head | 900 | physical_static | 0.148370 | 0.156068 | 0.007698 | 0.001551 | 0.006148 |
| train | head | 900 | tail_reversed | 0.153435 | 0.160493 | 0.007058 | 0.000910 | 0.006148 |
| train | wrists | 50 | paired | 0.011958 | 0.012111 | 0.000154 | 0.000003 | 0.000150 |
| train | wrists | 50 | physical_static | 0.016983 | 0.017101 | 0.000118 | -0.000032 | 0.000150 |
| train | wrists | 50 | tail_reversed | 0.034599 | 0.034560 | -0.000040 | -0.000190 | 0.000150 |
| train | wrists | 250 | paired | 0.013849 | 0.014034 | 0.000186 | 0.000006 | 0.000179 |
| train | wrists | 250 | physical_static | 0.019932 | 0.020064 | 0.000132 | -0.000047 | 0.000179 |
| train | wrists | 250 | tail_reversed | 0.037278 | 0.037236 | -0.000042 | -0.000221 | 0.000179 |
| train | wrists | 500 | paired | 0.020675 | 0.020926 | 0.000252 | -0.000072 | 0.000324 |
| train | wrists | 500 | physical_static | 0.030967 | 0.031085 | 0.000118 | -0.000206 | 0.000324 |
| train | wrists | 500 | tail_reversed | 0.047572 | 0.047491 | -0.000082 | -0.000405 | 0.000324 |
| train | wrists | 900 | paired | 0.089322 | 0.095032 | 0.005710 | 0.000074 | 0.005636 |
| train | wrists | 900 | physical_static | 0.159021 | 0.166719 | 0.007697 | 0.002062 | 0.005636 |
| train | wrists | 900 | tail_reversed | 0.174965 | 0.183254 | 0.008289 | 0.002654 | 0.005636 |
| val | head | 50 | paired | 0.009871 | 0.009857 | -0.000014 | -0.000089 | 0.000075 |
| val | head | 50 | physical_static | 0.011266 | 0.011204 | -0.000062 | -0.000137 | 0.000075 |
| val | head | 50 | tail_reversed | 0.009507 | 0.009503 | -0.000004 | -0.000079 | 0.000075 |
| val | head | 250 | paired | 0.011840 | 0.011821 | -0.000019 | -0.000114 | 0.000095 |
| val | head | 250 | physical_static | 0.013153 | 0.013069 | -0.000084 | -0.000179 | 0.000095 |
| val | head | 250 | tail_reversed | 0.011385 | 0.011391 | 0.000006 | -0.000089 | 0.000095 |
| val | head | 500 | paired | 0.019193 | 0.019073 | -0.000120 | -0.000307 | 0.000187 |
| val | head | 500 | physical_static | 0.020545 | 0.020327 | -0.000218 | -0.000405 | 0.000187 |
| val | head | 500 | tail_reversed | 0.018985 | 0.018934 | -0.000051 | -0.000238 | 0.000187 |
| val | head | 900 | paired | 0.084448 | 0.084126 | -0.000322 | -0.003861 | 0.003539 |
| val | head | 900 | physical_static | 0.085091 | 0.085130 | 0.000038 | -0.003501 | 0.003539 |
| val | head | 900 | tail_reversed | 0.084493 | 0.083937 | -0.000556 | -0.004095 | 0.003539 |
| val | wrists | 50 | paired | 0.020061 | 0.020000 | -0.000060 | -0.000136 | 0.000075 |
| val | wrists | 50 | physical_static | 0.011265 | 0.011190 | -0.000075 | -0.000150 | 0.000075 |
| val | wrists | 50 | tail_reversed | 0.019835 | 0.019772 | -0.000063 | -0.000138 | 0.000075 |
| val | wrists | 250 | paired | 0.022326 | 0.022171 | -0.000155 | -0.000252 | 0.000097 |
| val | wrists | 250 | physical_static | 0.013461 | 0.013284 | -0.000177 | -0.000274 | 0.000097 |
| val | wrists | 250 | tail_reversed | 0.022152 | 0.021991 | -0.000161 | -0.000259 | 0.000097 |
| val | wrists | 500 | paired | 0.029222 | 0.029282 | 0.000060 | -0.000151 | 0.000211 |
| val | wrists | 500 | physical_static | 0.020509 | 0.020416 | -0.000093 | -0.000304 | 0.000211 |
| val | wrists | 500 | tail_reversed | 0.029093 | 0.029142 | 0.000049 | -0.000162 | 0.000211 |
| val | wrists | 900 | paired | 0.094116 | 0.096937 | 0.002822 | 0.000014 | 0.002808 |
| val | wrists | 900 | physical_static | 0.079504 | 0.082286 | 0.002782 | -0.000026 | 0.002808 |
| val | wrists | 900 | tail_reversed | 0.093407 | 0.096289 | 0.002882 | 0.000074 | 0.002808 |

## Trace：全部实际S1条件/t的局部文本增量

不是旧guidance轨迹，没有自由输出因果归因或新target MSE。

| split | config | condition | t | text increment RMS | min(source RMS) | max(source RMS) |
|---|---|---|---:|---:|---:|---:|
| train | head | paired | 50 | 0.012851 | 0.010195 | 0.015368 |
| train | head | paired | 250 | 0.013386 | 0.010145 | 0.015836 |
| train | head | paired | 500 | 0.017707 | 0.013660 | 0.021411 |
| train | head | paired | 900 | 0.085531 | 0.048529 | 0.113452 |
| train | head | physical_static | 50 | 0.012445 | 0.009454 | 0.015581 |
| train | head | physical_static | 250 | 0.013205 | 0.009927 | 0.016255 |
| train | head | physical_static | 500 | 0.016931 | 0.013039 | 0.020870 |
| train | head | physical_static | 900 | 0.090203 | 0.062252 | 0.120233 |
| train | head | text_only | 50 | 0.012445 | 0.009454 | 0.015581 |
| train | head | text_only | 250 | 0.013205 | 0.009927 | 0.016255 |
| train | head | text_only | 500 | 0.016931 | 0.013039 | 0.020870 |
| train | head | text_only | 900 | 0.090203 | 0.062252 | 0.120233 |
| train | wrists | paired | 50 | 0.013141 | 0.010664 | 0.016865 |
| train | wrists | paired | 250 | 0.014120 | 0.011454 | 0.018573 |
| train | wrists | paired | 500 | 0.018782 | 0.014165 | 0.026784 |
| train | wrists | paired | 900 | 0.087286 | 0.065959 | 0.115823 |
| train | wrists | physical_static | 50 | 0.011339 | 0.008539 | 0.013987 |
| train | wrists | physical_static | 250 | 0.012413 | 0.009098 | 0.015835 |
| train | wrists | physical_static | 500 | 0.016796 | 0.012360 | 0.022914 |
| train | wrists | physical_static | 900 | 0.091778 | 0.060470 | 0.132149 |
| train | wrists | text_only | 50 | 0.011339 | 0.008539 | 0.013987 |
| train | wrists | text_only | 250 | 0.012413 | 0.009098 | 0.015835 |
| train | wrists | text_only | 500 | 0.016796 | 0.012360 | 0.022914 |
| train | wrists | text_only | 900 | 0.091778 | 0.060470 | 0.132149 |
| val | head | backbone_bypass | 50 | 0.008519 | 0.007131 | 0.010556 |
| val | head | backbone_bypass | 250 | 0.009053 | 0.007568 | 0.011537 |
| val | head | backbone_bypass | 500 | 0.012333 | 0.009001 | 0.017970 |
| val | head | backbone_bypass | 900 | 0.065031 | 0.037149 | 0.084207 |
| val | head | imu_only | 50 | 0.000000 | 0.000000 | 0.000000 |
| val | head | imu_only | 250 | 0.000000 | 0.000000 | 0.000000 |
| val | head | imu_only | 500 | 0.000000 | 0.000000 | 0.000000 |
| val | head | imu_only | 900 | 0.000000 | 0.000000 | 0.000000 |
| val | head | numeric_zero | 50 | 0.008299 | 0.006776 | 0.009731 |
| val | head | numeric_zero | 250 | 0.009060 | 0.007048 | 0.010931 |
| val | head | numeric_zero | 500 | 0.011972 | 0.008568 | 0.015603 |
| val | head | numeric_zero | 900 | 0.056272 | 0.042550 | 0.078715 |
| val | head | paired | 50 | 0.008103 | 0.006718 | 0.010533 |
| val | head | paired | 250 | 0.008839 | 0.007339 | 0.011572 |
| val | head | paired | 500 | 0.012268 | 0.009193 | 0.018066 |
| val | head | paired | 900 | 0.063457 | 0.039236 | 0.084011 |
| val | head | physical_static | 50 | 0.008519 | 0.007131 | 0.010556 |
| val | head | physical_static | 250 | 0.009053 | 0.007568 | 0.011537 |
| val | head | physical_static | 500 | 0.012333 | 0.009001 | 0.017970 |
| val | head | physical_static | 900 | 0.065031 | 0.037149 | 0.084207 |
| val | head | tail_reversed | 50 | 0.008030 | 0.006689 | 0.010326 |
| val | head | tail_reversed | 250 | 0.008758 | 0.007166 | 0.011554 |
| val | head | tail_reversed | 500 | 0.012028 | 0.009017 | 0.017780 |
| val | head | tail_reversed | 900 | 0.064280 | 0.039566 | 0.081840 |
| val | head | text_only | 50 | 0.008519 | 0.007131 | 0.010556 |
| val | head | text_only | 250 | 0.009053 | 0.007568 | 0.011537 |
| val | head | text_only | 500 | 0.012333 | 0.009001 | 0.017970 |
| val | head | text_only | 900 | 0.065031 | 0.037149 | 0.084207 |
| val | wrists | backbone_bypass | 50 | 0.007511 | 0.006562 | 0.009059 |
| val | wrists | backbone_bypass | 250 | 0.008053 | 0.007204 | 0.009495 |
| val | wrists | backbone_bypass | 500 | 0.010522 | 0.009678 | 0.011446 |
| val | wrists | backbone_bypass | 900 | 0.082966 | 0.050227 | 0.115663 |
| val | wrists | imu_only | 50 | 0.000000 | 0.000000 | 0.000000 |
| val | wrists | imu_only | 250 | 0.000000 | 0.000000 | 0.000000 |
| val | wrists | imu_only | 500 | 0.000000 | 0.000000 | 0.000000 |
| val | wrists | imu_only | 900 | 0.000000 | 0.000000 | 0.000000 |
| val | wrists | numeric_zero | 50 | 0.008776 | 0.008273 | 0.009198 |
| val | wrists | numeric_zero | 250 | 0.009336 | 0.008987 | 0.009549 |
| val | wrists | numeric_zero | 500 | 0.011896 | 0.011100 | 0.013153 |
| val | wrists | numeric_zero | 900 | 0.072193 | 0.059214 | 0.085374 |
| val | wrists | paired | 50 | 0.008591 | 0.006830 | 0.011979 |
| val | wrists | paired | 250 | 0.008954 | 0.007285 | 0.011968 |
| val | wrists | paired | 500 | 0.011144 | 0.009418 | 0.013638 |
| val | wrists | paired | 900 | 0.076425 | 0.051805 | 0.101036 |
| val | wrists | physical_static | 50 | 0.007511 | 0.006562 | 0.009059 |
| val | wrists | physical_static | 250 | 0.008053 | 0.007204 | 0.009495 |
| val | wrists | physical_static | 500 | 0.010522 | 0.009678 | 0.011446 |
| val | wrists | physical_static | 900 | 0.082966 | 0.050227 | 0.115663 |
| val | wrists | tail_reversed | 50 | 0.008553 | 0.006832 | 0.011971 |
| val | wrists | tail_reversed | 250 | 0.008902 | 0.007308 | 0.011861 |
| val | wrists | tail_reversed | 500 | 0.010983 | 0.009416 | 0.013446 |
| val | wrists | tail_reversed | 900 | 0.076672 | 0.051915 | 0.101739 |
| val | wrists | text_only | 50 | 0.007511 | 0.006562 | 0.009059 |
| val | wrists | text_only | 250 | 0.008053 | 0.007204 | 0.009495 |
| val | wrists | text_only | 500 | 0.010522 | 0.009678 | 0.011446 |
| val | wrists | text_only | 900 | 0.082966 | 0.050227 | 0.115663 |
