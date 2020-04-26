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
	int answer; //�������ꊇ���I�΂���Ƃ���
	int input = 3;//���̓f�[�^�̐�
	int output = 1;//�o�̓f�[�^�̐�
	int data = 6;//�f�[�^��
	int middle_layer;//���ԑw�̐�
	int each_layer;//�e�w�̑f�q��
	double data_mi[][3] = { {0,1,0},{0,1,1} };
	double** x_input;//���̓f�[�^�̔z��
	double* y_output;//�o�̓f�[�^�̔z��
	double epsilon = 0.05;//�w�K��
	double*** w;//�d�݁A������w�̏ォ��̏��ԁA���̑w�̏ォ��̏��ԁA���w�ڂ���\��

	//�w���A�f�q���̎w��
	printf("���ԑw�̐����w�肵�Ă�������\n");
	scanf_s("%d", &middle_layer);
	printf("�e�w�̑f�q�����w�肵�Ă�������\n");
	scanf_s("%d", &each_layer);

	//���I�ɓ��̓f�[�^��錾
	x_input = (double**)malloc(sizeof(double*) * data); //double*�^�̃X�y�[�X��v�f���i[��][]�j�̕������m�ۂ���B
	for (i = 0; i < data; i++)
	{
		x_input[i] = (double*)malloc(sizeof(double) * input); //double�^�̃X�y�[�X��v�f���i[][��]�j�̕������m�ۂ���B
	}
	//���I�ɏo�̓f�[�^��錾
	y_output = (double*)malloc(sizeof(double) * data); //double*�^�̃X�y�[�X��v�f���i[��][]�j�̕������m�ۂ���B
	/*for (i = 0; i < data; i++)
	{
		y_output[i] = (double*)malloc(sizeof(double) * output); //double�^�̃X�y�[�X��v�f���i[][��]�j�̕������m�ۂ���B
	}*/

	//�t�@�C�����J���A�f�[�^���i�[����
	FILE* fp;
	errno_t error;
	error = fopen_s(&fp, "test.csv", "r");

	if (error != 0)
		printf("�t�@�C�����J���܂���ł���");
	else {
		for (i = 0; i < data; i++) {
			for (j = 0; j < input; j++) {
				fscanf_s(fp, "%lf, ", &x_input[i][j]);
			}
				if (fscanf_s(fp, "%lf, ", &y_output[i]) != '\0');
		}
	}
	//�m�F�p
	/*for (i = 0; i < data; i++) {
		for (j = 0; j < input; j++) {
			printf("%f\n", x_input[i][j]);
		}
			printf("%f\n", y_output[i]);
	}*/
	//�d�݂̔z����m�ۂ���������
	w = (double***)malloc(sizeof(double**) * (each_layer+input));//���͎��� > �e�w�̑f�q���̎��p�ɑ��߂Ɋm��
	for (i = 0; i < each_layer + input; i++)
	{
		w[i] = (double**)malloc(sizeof(double*) * (each_layer+output));
		for (j = 0; j < each_layer+output; j++) {
			w[i][j] = (double*)malloc(sizeof(double) * middle_layer+1);
		}
	}

	srand(time(NULL));
	for (i = 0;i < each_layer + input;i++) {//���͎���>�e�w�̑f�q���̎��p�ɑ��߂Ɋi�[
		for (j = 0;j < each_layer+output;j++) {//�o�͎���>�e�w�̑f�q���̎��̂��߂ɑ��߂�
			for (k = 0;k < middle_layer + 1;k++) {
				w[i][j][k] = (double)rand() / RAND_MAX;
				//printf("%lf\n", w2[i][j][k]);
			}
		}
	}
	
	//�ꊇ��������
	printf("�����w�K 1 �ꊇ�w�K 2\n");
	scanf_s("%d", &answer);
	if (answer == 1) {
		online_training(input, output, data, each_layer, middle_layer, x_input, y_output, w, epsilon);
	}
	if (answer == 2) {
		batch_training(input, output, data, each_layer, middle_layer, x_input, y_output, w, epsilon);
	}
	return 0;
}
//�����w�K
void online_training(int input, int output, int data, int each_layer, int middle_layer, double** x_input, double* y_output, double*** w, double epsilon)
{
	int i, j, k,l;
	int cnt = 0;//�J�E���g�p
	double I[10][10] = { 0 };//�������̓��͔z��A�����ォ��̏��ԁA�E���w�̔ԍ�
	double sum_u[10][10] = { 0 };//�o�́��d�݁{�o�C�A�X���󂯎��z��
	double sum_r[10][10] = { 0 };//�t�`�d�̍ۂ̔z��@�����ォ��̏��ԁA�E���w�̐�
	double Y_in[10] = { 0 };//�o�͑w�̓���
	double Y_out[10] = { 0 };//�o�͑w�̏o��
	double E[10] = { 0 };//�덷���i�[����z��

	do {
		for (l = 0;l < data;l++) {
			//������
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
			//1�w�ڂ̓��͔z��ɓ��̓f�[�^����
			for (i = 0;i < input;i++) {
				I[i][0] = x_input[l][i];
			}
			//2�w�ڂ̓���
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < input + 1;j++) {
					if (j == input)
						sum_u[i][0] += w[j][i][0];//�o�C�A�X
					else
						sum_u[i][0] += I[j][0] * w[j][i][0];
				}
				I[i][1] = sigmoid(sum_u[i][0]);
			}

			if (middle_layer == 1) {//���ԑw��1�̂Ƃ�
				for (i = 0;i < output;i++) {
					for (j = 0;j < each_layer + 1;j++) {
						if (j == each_layer)
							Y_in[i] += w[j][i][1];//�o�C�A�X
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
								sum_u[j][i] += w[k][j][i];//�o�C�A�X
							else
								sum_u[j][i] += I[k][i] * w[k][j][i];
						}
						I[j][i + 1] = sigmoid(sum_u[j][i]);
					}
				}
			}
			k = middle_layer;
			//�o�͑w�̑O�܂ŗ�����
			for (i = 0;i < output;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						Y_in[i] += w[j][i][k];//�o�C�A�X
					else
						Y_in[i] += I[j][k] * w[j][i][k];
				}
				Y_out[i] = sigmoid(Y_in[i]);
			}
			//�덷�����߂�B
			for (i = 0;i < output;i++) {
				E[0] += pow(Y_out[i] - y_output[l], 2) / 2.0;
			}
			//printf("%lf\n", E[0]);
			//��������덷�t�`�d�A�o�͑w�Ɍ������d�݂̍X�V
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
			//���ԑw��1�̏ꍇ
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
				//���ԑw��2�̂Ƃ�
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
			}//���ԑw��2��葽����
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
	printf("�w�K�̏I��\n");
	printf("count = %d  error = %f  Y = %f\n", cnt, E[0], Y_out[0]);
	if (cnt == 50000) {
		printf("count = %d\n", cnt);
	}
	else if (E[0] <= 0.00001) {
		printf("�w�K���I���܂���\n");

		printf("error = %f\n", E[0]);
	}
	return 0;
}
//�ꊇ�w�K
void batch_training(int input, int output, int data, int each_layer, int middle_layer, double** x_input, double* y_output, double*** w, double epsilon)
{
	int i, j, k, l;
	int cnt = 0;//�J�E���g�p
	double I[10][10][10] = { 0 };//�������̓��͔z��A�����f�[�^�̏��ԁA�^�񒆂��f�q�̏ォ��̏��ԁA�E���w�̔ԍ�
	double I_ave[10][10] = { 0 };//�O�̑w�̏o�͕��ς��i�[����
	double sum_u[10][10][10] = { 0 };//�o�́��d�݁{�o�C�A�X���󂯎��z��
	double sum_r[10][10] = { 0 };//�t�`�d�̍ۂ̔z��@
	double Y_in[10][10] = { 0 };//�o�͑w�̓���
	double Y_out[10][10] = { 0 };//�o�͑w�̏o��
	double E[10] = { 0 };//�덷
	double sum_E = 0;//�덷�̍��v

	do {
		sum_E = 0;//�덷�̍��v�̏�����

		for (l = 0;l < data;l++) {
			//������
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

			//1�w�ڂ̓��͔z��ɓ��̓f�[�^����
			for (i = 0;i < input;i++) {
				I[l][i][0] = x_input[l][i];
			}
			//2�w�ڂ̓���
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < input + 1;j++) {
					if (j == input)
						sum_u[l][i][0] += w[j][i][0];//�o�C�A�X
					else
						sum_u[l][i][0] += I[l][j][0] * w[j][i][0];
				}
				I[l][i][1] = sigmoid(sum_u[l][i][0]);
			}

			if (middle_layer == 1) {//���ԑw��1�̂Ƃ�
				for (i = 0;i < output;i++) {
					for (j = 0;j < each_layer + 1;j++) {
						if (j == each_layer)
							Y_in[l][i] += w[j][i][1];//�o�C�A�X
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
								sum_u[l][j][i] += w[k][j][i];//�o�C�A�X
							else
								sum_u[l][j][i] += I[l][k][i] * w[k][j][i];
						}
						I[l][j][i + 1] = sigmoid(sum_u[l][j][i]);
					}
				}
			}
			k = middle_layer;
			//�o�͑w�̑O�܂ŗ�����
			for (i = 0;i < output;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						Y_in[l][i] += w[j][i][k];//�o�C�A�X
					else
						Y_in[l][i] += I[l][j][k] * w[j][i][k];
				}
				Y_out[l][i] = sigmoid(Y_in[l][i]);
				printf("Y=%lf ", Y_out[l][i]);
			}
			//�덷�����߂�B
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
		//��������덷�t�`�d
		//�f�[�^���������ĕ��ς��Ƃ�
		for (i = 0;i < output;i++) {
			for (l = 0;l < data;l++) {
				sum_r[i][k] += (Y_out[l][i] - y_output[l]) * Y_out[l][i] * (1 - Y_out[l][i]);
			}
			sum_r[i][k] = sum_r[i][k] / data;
		}
		//�O�̑w�̏o�͂̕��ς����߂�
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
		//���ԑw��1�̏ꍇ
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
			//�O�̑w�̏o�͂̕��ς����߂�
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
			//���ԑw��2�̂Ƃ�
			k = middle_layer - 1;
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < output;j++) {
					for (l = 0;l < data;l++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
					}
				}
				sum_r[i][k] = sum_r[i][k] / data;
			}
			//�O�̑w�̏o�͂̕��ς����߂�
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
			//�O�̑w�̏o�͂̕��ς����߂�
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
		//���ԑw��2��葽����
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
			//�O�̑w�̏o�͂̕��ς����߂�
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
				//�O�̑w�̏o�͂̕��ς����߂�
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
			//���͑wk=0
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < each_layer;j++) {
					for (l = 0;l < data;l++) {
						sum_r[i][k] += sum_r[j][k + 1] * w[i][j][k + 1] * I[l][i][k + 1] * (1 - I[l][i][k + 1]);
					}
				}
				sum_r[i][k] = sum_r[i][k] / data;
			}
			//�O�̑w�̏o�͂̕��ς����߂�
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
	printf("�w�K�̏I��\n");
	printf("count = %d  error = %f \n", cnt, sum_E);
	if (cnt == 50000) {
		printf("count = %d\n", cnt);
	}
	else if (sum_E <= 0.00001) {
		printf("�w�K���I���܂���\n");

		printf("error = %f\n", sum_E);
	}
	return 0;
}
//�V�O���C�h�֐�
double sigmoid(double s) {
	return 1 / (1 + exp(-s));
}
void migaku(int input, int output, int data, int each_layer, double **x_input, int middle_layer, double*** w)
{
	int i, j, k, l;
	double sum_u[10][10][10];//�������̃V�O���C�h�֐��ɓ����O�̘a �����f�q���A�E���w��
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

	//�f�[�^���ƂɊw�K
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



		//���ԑw����̃m�[�h���̏o�͂̍��v�����߂�B
		for (k = 1;k < middle_layer;k++) {
			for (i = 0;i < each_layer;i++) {
				for (j = 0;j < each_layer + 1;j++) {
					if (j == each_layer)
						sum_u[i][k][l] = sum_u[i][k][l] + w[j][i][k];
					else
						sum_u[i][k][l] = sum_u[i][k][l] + sigmoid(sum_u[j][k - 1][l]) * w[j][i][k];//�V�O���C�h�ɓ��ꂽ�o�͂Əd�݂̐ς̍��v
						//printf("%lf\n", sum_u[i][k]);
				}
				//printf("%lf\n", sum_u[i][k]);
			}
		}

		//printf("%lf %lf %lf\n", sum_u[0][2],sum_u[1][2],sum_u[2][2]);
		//���v�����߂�

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