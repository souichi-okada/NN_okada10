#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>

void online_training(int input, int output, int data, int each_layer, int middle_layer, double** x_input, double* y_output, double*** w, double epsilon);
void batch_training(int input, int output, int data, int each_layer, int middle_layer, double** x_input, double* y_output, double*** w, double epsilon);
double Error(double y, double* data2);
void migaku(int input, int output, int data, int each_layer, double** x_input, int middle_layer, double*** w);
double sigmoid(double s);

int main()
{
	int i, j, k;
	int answer; //逐次か一括か選ばせるときの
	int input = 3;//入力データの数
	int output = 1;//出力データの数
	int data = 6;//データ数
	int middle_layer;//中間層の数
	int each_layer;//各層の素子数
	double data_mi[][3] = { {0,1,0},{0,1,1} };
	double** x_input;//入力データの配列
	double* y_output;//出力データの配列
	double epsilon = 0.05;//学習率
	double*** w;//重み、左から層の上からの順番、次の層の上からの順番、何層目かを表す

	//層数、素子数の指定
	printf("中間層の数を指定してください\n");
	scanf_s("%d", &middle_layer);
	printf("各層の素子数を指定してください\n");
	scanf_s("%d", &each_layer);

	//動的に入力データを宣言
	x_input = (double**)malloc(sizeof(double*) * data); //double*型のスペースを要素数（[☆][]）の分だけ確保する。
	for (i = 0; i < data; i++)
	{
		x_input[i] = (double*)malloc(sizeof(double) * input); //double型のスペースを要素数（[][☆]）の分だけ確保する。
	}
	//動的に出力データを宣言
	y_output = (double*)malloc(sizeof(double) * data); //double*型のスペースを要素数（[☆][]）の分だけ確保する。
	/*for (i = 0; i < data; i++)
	{
		y_output[i] = (double*)malloc(sizeof(double) * output); //double型のスペースを要素数（[][☆]）の分だけ確保する。
	}*/

	//ファイルを開き、データを格納する
	FILE* fp;
	errno_t error;
	error = fopen_s(&fp, "test.csv", "r");

	if (error != 0)
		printf("ファイルを開けませんでした");
	else {
		for (i = 0; i < data; i++) {
			for (j = 0; j < input; j++) {
				fscanf_s(fp, "%lf, ", &x_input[i][j]);
			}
				if (fscanf_s(fp, "%lf, ", &y_output[i]) != '\0');
		}
	}
	//確認用
	/*for (i = 0; i < data; i++) {
		for (j = 0; j < input; j++) {
			printf("%f\n", x_input[i][j]);
		}
			printf("%f\n", y_output[i]);
	}*/
	//重みの配列を確保し乱数を代入
	w = (double***)malloc(sizeof(double**) * (each_layer+input));//入力次元 > 各層の素子数の時用に多めに確保
	for (i = 0; i < each_layer + input; i++)
	{
		w[i] = (double**)malloc(sizeof(double*) * (each_layer+output));
		for (j = 0; j < each_layer+output; j++) {
			w[i][j] = (double*)malloc(sizeof(double) * middle_layer+1);
		}
	}

	srand(time(NULL));
	for (i = 0;i < each_layer + input;i++) {//入力次元>各層の素子数の時用に多めに格納
		for (j = 0;j < each_layer+output;j++) {//出力次元>各層の素子数の時のために多めに
			for (k = 0;k < middle_layer + 1;k++) {
				w[i][j][k] = (double)rand() / RAND_MAX;
				//printf("%lf\n", w2[i][j][k]);
			}
		}
	}
	
	//一括か逐次か
	printf("逐次学習 1 一括学習 2\n");
	scanf_s("%d", &answer);
	if (answer == 1) {
		online_training(input, output, data, each_layer, middle_layer, x_input, y_output, w, epsilon);
	}
	if (answer == 2) {
		batch_training(input, output, data, each_layer, middle_layer, x_input, y_output, w, epsilon);
	}
	return 0;
}
//逐次学習
void online_training(int input, int output, int data, int each_layer, int middle_layer, double** x_input, double* y_output, double*** w, double epsilon)
{
	int i, j, k,l;
	int cnt = 0;//カウント用
	double I[10][10] = { 0 };//順方向の入力配列、左が上からの順番、右が層の番号
	double sum_u[10][10] = { 0 };//出力＊重み＋バイアスを受け取る配列
	double sum_r[10][10] = { 0 };//逆伝播の際の配列　左が上からの順番、右が層の数
	double Y_in[10] = { 0 };//出力層の入力
	double Y_out[10] = { 0 };//出力層の出力
	double E[10] = { 0 };//誤差を格納する配列

	do {
		for (l = 0;l < data;l++) {
			//初期化
			for (i = 0;i < 10;i++) {
				for (j = 0;j < 10;j++) {
					sum_u[i][j] = 0;
					sum_r[i][j] = 0;
					I[i][j] = 0;
				}
			}
			for (i = 0;i < 10;i++) {
				Y_in[i] = 0;
				Y_out[i] = 0;
				E[i] = 0;
			}
			//1層目の入力配列に入力データを代入
			for (i = 0;i < input;i++) {
				I[i][0] = x_input[l][i];
			}
			//2層目の入力
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < input + 1;j++) {
					if (j == input)
						sum_u[i][0] += w[j][i][0];//バイアス
					else
						sum_u[i][0] += I[j][0] * w[j][i][0];
				}
				I[i][1] = sigmoid(sum_u[i][0]);
			}

			if (middle_layer == 1) {//中間層が1のとき
				for (i = 0;i < output;i++) {
					for (j = 0;j < each_layer + 1;j++) {
						if (j == each_layer)
							Y_in[i] += w[j][i][1];//バイアス
						else
							Y_in[i] += I[j][1] * w[j][i][1];
					}
					Y_out[i] = sigmoid(Y_in[i]);
				}
				for (i = 0;i < output;i++) {
					E[0] += pow(Y_out[i] - y_output[l], 2) / 2.0;
				}
			}
			else {
				for (i = 1;i < middle_layer;i++) {
					for (j = 0;j < each_layer;j++) {
						for (k = 0;k < each_layer + 1;k++) {
							if (k == each_layer)
								sum_u[j][i] += w[k][j][i];//バイアス
							else
								sum_u[j][i] += I[k][i] * w[k][j][i];
						}
						I[j][i + 1] = sigmoid(sum_u[j][i]);
					}
				}
			}
			k = middle_layer;
			//出力層の前まで来たら
			for (i = 0;i < output;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						Y_in[i] += w[j][i][k];//バイアス
					else
						Y_in[i] += I[j][k] * w[j][i][k];
				}
				Y_out[i] = sigmoid(Y_in[i]);
			}
			//誤差を求める。
			for (i = 0;i < output;i++) {
				E[0] += pow(Y_out[i] - y_output[l], 2) / 2.0;
			}
			//printf("%lf\n", E[0]);
			//ここから誤差逆伝播、出力層に向かう重みの更新
			//k = middle_layer;
			for (i = 0;i < output;i++) {
				sum_r[i][k] = (Y_out[i] - y_output[l])*Y_out[i] * (1 - Y_out[i]);
			}
			for (i = 0;i < output;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
					else
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I[j][k];
				}
			}
			//中間層が1つの場合
			if (middle_layer == 1) {
				k = middle_layer - 1;
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < output;j++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[i][k + 1] * (1 - I[i][k + 1]);
					}
				}
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < input + 1;j++) {
						if (j == input)
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
						else
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I[j][k];
					}
				}
			}
			else if (middle_layer == 2) {
				//中間層が2つのとき
				k = middle_layer - 1;
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < output;j++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[i][k + 1] * (1 - I[i][k + 1]);
					}
				}
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < each_layer + 1;j++) {
						if (j == each_layer)
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
						else
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I[j][k];
					}
				}
				k = middle_layer - 2;
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < each_layer;j++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[i][k + 1] * (1 - I[i][k + 1]);
					}
				}
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < input + 1;j++) {
						if (j == input)
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
						else
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I[j][k];
					}
				}
			}//中間層が2より多い時
			else if (middle_layer > 2) {
				k = middle_layer - 1;
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < output;j++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[i][k + 1] * (1 - I[i][k + 1]);
					}
				}
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < each_layer + 1;j++) {
						if (j == each_layer)
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
						else
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I[j][k];
					}
				}
				for (k = middle_layer - 2;k > 0;k--) {
					for (i = 0;i < each_layer;i++) {
						for (j = 0;j < each_layer;j++) {
							sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[i][k + 1] * (1 - I[i][k + 1]);
						}
					}
					for (i = 0;i < each_layer;i++) {
						for (j = 0;j < each_layer + 1;j++) {
							if (j == each_layer)
								w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
							else
								w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I[j][k];
						}
					}
				}
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < each_layer;j++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[i][k + 1] * (1 - I[i][k + 1]);
					}
				}
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < input + 1;j++) {
						if (j == input)
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
						else
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I[j][k];
					}
				}
			}
			cnt++;
			printf("count = %d  error = %f  Y = %f  t = %f\n", cnt, E[0],Y_out[0],y_output[l]);
		}	
	}while (E[0] > 0.0001 && cnt < 10000000);
	printf("学習の終了\n");
	printf("count = %d  error = %f  Y = %f\n", cnt, E[0], Y_out[0]);
	if (cnt == 50000) {
		printf("count = %d\n", cnt);
	}
	else if (E[0] <= 0.00001) {
		printf("学習を終えました\n");

		printf("error = %f\n", E[0]);
	}
	return 0;
}
//一括学習
void batch_training(int input, int output, int data, int each_layer, int middle_layer, double** x_input, double* y_output, double*** w, double epsilon)
{
	int i, j, k, l;
	int cnt = 0;//カウント用
	double I[10][10][10] = { 0 };//順方向の入力配列、左がデータの順番、真ん中が素子の上からの順番、右が層の番号
	double I_ave[10][10] = { 0 };//前の層の出力平均を格納する
	double sum_u[10][10][10] = { 0 };//出力＊重み＋バイアスを受け取る配列
	double sum_r[10][10] = { 0 };//逆伝播の際の配列　
	double Y_in[10][10] = { 0 };//出力層の入力
	double Y_out[10][10] = { 0 };//出力層の出力
	double E[10] = { 0 };//誤差
	double sum_E = 0;//誤差の合計

	do {
		sum_E = 0;//誤差の合計の初期化

		for (l = 0;l < data;l++) {
			//初期化
			for (i = 0;i < 10;i++) {
				for (j = 0;j < 10;j++) {
					for (k = 0;k < 10;k++) {
						sum_u[i][j][k] = 0;
						I[i][j][k] = 0;
					}
				}
			}
			for (i = 0;i < 10;i++) {
				for (j = 0;j < 10;j++) {
					Y_in[i][j] = 0;
					Y_out[i][j] = 0;
					sum_r[i][j] = 0;
					I_ave[i][j] = 0;
				}
			}
			for (i = 0;i < 10;i++) {
				E[i] = 0;
			}

			//1層目の入力配列に入力データを代入
			for (i = 0;i < input;i++) {
				I[l][i][0] = x_input[l][i];
			}
			//2層目の入力
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < input + 1;j++) {
					if (j == input)
						sum_u[l][i][0] += w[j][i][0];//バイアス
					else
						sum_u[l][i][0] += I[l][j][0] * w[j][i][0];
				}
				I[l][i][1] = sigmoid(sum_u[l][i][0]);
			}

			if (middle_layer == 1) {//中間層が1のとき
				for (i = 0;i < output;i++) {
					for (j = 0;j < each_layer + 1;j++) {
						if (j == each_layer)
							Y_in[l][i] += w[j][i][1];//バイアス
						else
							Y_in[l][i] += I[l][j][1] * w[j][i][1];
					}
					Y_out[l][i] = sigmoid(Y_in[l][i]);
				}
				for (i = 0;i < output;i++) {
					E[l] += pow(Y_out[l][i] - y_output[l], 2) / 2.0;
				}
			}
			else {
				for (i = 1;i < middle_layer;i++) {
					for (j = 0;j < each_layer;j++) {
						for (k = 0;k < each_layer + 1;k++) {
							if (k == each_layer)
								sum_u[l][j][i] += w[k][j][i];//バイアス
							else
								sum_u[l][j][i] += I[l][k][i] * w[k][j][i];
						}
						I[l][j][i + 1] = sigmoid(sum_u[l][j][i]);
					}
				}
			}
			k = middle_layer;
			//出力層の前まで来たら
			for (i = 0;i < output;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						Y_in[l][i] += w[j][i][k];//バイアス
					else
						Y_in[l][i] += I[l][j][k] * w[j][i][k];
				}
				Y_out[l][i] = sigmoid(Y_in[l][i]);
				printf("Y=%lf ", Y_out[l][i]);
			}
			//誤差を求める。
			for (i = 0;i < output;i++) {
				E[l] += pow(Y_out[l][i] - y_output[l], 2) / 2.0;
			}
		}
		for (i = 0;i < data;i++) {
			sum_E += E[i];
		}
		sum_E = sum_E / 6.0;
		//printf("%lf\n", sum_E);
		k = middle_layer;
		//ここから誤差逆伝播
		//データ数分足して平均をとる
		for (i = 0;i < output;i++) {
			for (l = 0;l < data;l++) {
				sum_r[i][k] += (Y_out[l][i] - y_output[l]) * Y_out[l][i] * (1 - Y_out[l][i]);
			}
			sum_r[i][k] = sum_r[i][k] / data;
		}
		//前の層の出力の平均を求める
		for (i = 0;i < each_layer;i++) {
			for (l = 0;l < data;l++) {
				I_ave[i][k] += I[l][i][k];
			}
			I_ave[i][k] = I_ave[i][k] / data;
		}
		for (i = 0;i < output;i++) {
			for (j = 0;j < each_layer + 1;j++) {
				if (j == each_layer)
					w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
				else
					w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I_ave[j][k];
			}
		}
		//中間層が1つの場合
		if (middle_layer == 1) {
			k = middle_layer - 1;
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < output;j++) {
					for (l = 0;l < data;l++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
					}
				}
				sum_r[i][k] = sum_r[i][k] / data;
			}
			//前の層の出力の平均を求める
			for (i = 0;i < input;i++) {
				for (l = 0;l < data;l++) {
					I_ave[i][k] += I[l][i][k];
				}
				I_ave[i][k] = I_ave[i][k] / data;
			}
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < input + 1;j++) {
					if (j == input)
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
					else
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I_ave[j][k];
				}
			}
		}
		else if (middle_layer == 2) {
			//中間層が2つのとき
			k = middle_layer - 1;
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < output;j++) {
					for (l = 0;l < data;l++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
					}
				}
				sum_r[i][k] = sum_r[i][k] / data;
			}
			//前の層の出力の平均を求める
			for (i = 0;i < each_layer;i++) {
				for (l = 0;l < data;l++) {
					I_ave[i][k] += I[l][i][k];
				}
				I_ave[i][k] = I_ave[i][k] / data;
			}
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
					else
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I_ave[j][k];
				}
			}
			k = middle_layer - 2;
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < each_layer;j++) {
					for (l = 0;l < data;l++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
					}
				}
				sum_r[i][k] = sum_r[i][k] / data;
			}
			//前の層の出力の平均を求める
			for (i = 0;i < input;i++) {
				for (l = 0;l < data;l++) {
					I_ave[i][k] += I[l][i][k];
				}
				I_ave[i][k] = I_ave[i][k] / data;
			}
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < input + 1;j++) {
					if (j == input)
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
					else
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I_ave[j][k];
				}
			}
		}
		//中間層が2より多い時
		else if (middle_layer > 2) {
			k = middle_layer - 1;
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < output;j++) {
					for (l = 0;l < data;l++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
					}
				}
				sum_r[i][k] = sum_r[i][k] / data;
			}
			//前の層の出力の平均を求める
			for (i = 0;i < each_layer;i++) {
				for (l = 0;l < data;l++) {
					I_ave[i][k] += I[l][i][k];
				}
				I_ave[i][k] = I_ave[i][k] / data;
			}
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
					else
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I_ave[j][k];
				}
			}

			for (k = middle_layer - 2;k > 0;k--) {
				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < each_layer;j++) {
						for (l = 0;l < data;l++) {
							sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
						}
					}
					sum_r[i][k] = sum_r[i][k] / data;
				}
				//前の層の出力の平均を求める
				for (i = 0;i < each_layer;i++) {
					for (l = 0;l < data;l++) {
						I_ave[i][k] += I[l][i][k];
					}
					I_ave[i][k] = I_ave[i][k] / data;
				}

				for (i = 0;i < each_layer;i++) {
					for (j = 0;j < each_layer + 1;j++) {
						if (j == each_layer)
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
						else
							w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I_ave[j][k];
					}
				}
			}
			//入力層k=0
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < each_layer;j++) {
					for (l = 0;l < data;l++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
					}
				}
				sum_r[i][k] = sum_r[i][k] / data;
			}
			//前の層の出力の平均を求める
			for (i = 0;i < input;i++) {
				for (l = 0;l < data;l++) {
					I_ave[i][k] += I[l][i][k];
				}
				I_ave[i][k] = I_ave[i][k] / data;
			}
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < input + 1;j++) {
					if (j == input)
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k];
					else
						w[j][i][k] = w[j][i][k] - epsilon * sum_r[i][k] * I_ave[j][k];
				}
			}
		}
		cnt++;
		printf("count = %d  error = %f \n", cnt, sum_E);
	} while (sum_E > 0.0001 && cnt < 500000);
	printf("学習の終了\n");
	printf("count = %d  error = %f \n", cnt, sum_E);
	if (cnt == 50000) {
		printf("count = %d\n", cnt);
	}
	else if (sum_E <= 0.00001) {
		printf("学習を終えました\n");

		printf("error = %f\n", sum_E);
	}
	return 0;
}
//シグモイド関数
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}
void migaku(int input, int output, int data, int each_layer, double **x_input, int middle_layer, double*** w)
{
	int i, j, k, l;
	double sum_u[10][10][10];//順方向のシグモイド関数に入れる前の和 左が素子数、右が層数
	double y[10] = { 0 };

	for (i = 0;i < 10;i++) {
		y[i] = 0;
	}


	for (i = 0;i < 10;i++) {
		for (j = 0;j < 10;j++) {
			for (k = 0;k < 10;k++) {
				sum_u[i][j][k] = 0;
			}
		}
	}

	//データごとに学習
	for (l = 0;l < data;l++) {
		for (i = 0;i < each_layer;i++) {
			for (j = 0;j < input + 1;j++) {
				if (j == input)
					sum_u[i][0][l] = sum_u[i][0][l] + w[j][i][0];
				else
					sum_u[i][0][l] = sum_u[i][0][l] + x_input[l][j] * w[j][i][0];
			}
			//printf("%lf\n", sum_u[i][0]);
		}



		//中間層からのノード毎の出力の合計を求める。
		for (k = 1;k < middle_layer;k++) {
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						sum_u[i][k][l] = sum_u[i][k][l] + w[j][i][k];
					else
						sum_u[i][k][l] = sum_u[i][k][l] + sigmoid(sum_u[j][k - 1][l]) * w[j][i][k];//シグモイドに入れた出力と重みの積の合計
						//printf("%lf\n", sum_u[i][k]);
				}
				//printf("%lf\n", sum_u[i][k]);
			}
		}

		//printf("%lf %lf %lf\n", sum_u[0][2],sum_u[1][2],sum_u[2][2]);
		//合計を求める

		if (k == middle_layer) {
			for (i = 0;i < output;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						y[l] = y[l] + w[j][i][k];
					else
						y[l] = y[l] + sigmoid(sum_u[j][k - 1][l]) * w[j][i][k];
				}
			}
		}
	}
	y[l] = sigmoid(y[l]);
	printf("out = %f\n\n", y[l]);

}