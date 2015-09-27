
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <ctime>

#include <CL/cl.h>
#include <CL/opencl.h>

#define MSUCCEES 1
#define MFAIL 0

#define NX 2000
#define NY 2000
#define UNUM 1000

using namespace std;

cl_float matrix_0[NX][NY], matrix_1[NX][NY], matrix_2[NX][NY], matrix_3[NX][NY];
clock_t T1, T2, T3, T4;

int convertToString(const char *filename, std::string& s)
{
	size_t size;
	char*  str;
	std::fstream f(filename, (std::fstream::in | std::fstream::binary));

	if(f.is_open())
	{
		size_t fileSize;
		f.seekg(0, std::fstream::end);
		size = fileSize = (size_t)f.tellg();
		f.seekg(0, std::fstream::beg);
		str = new char[size+1];
		if(!str)
		{
			f.close();
			return 0;
		}

		f.read(str, fileSize);
		f.close();
		str[size] = '\0';
		s = str;
		delete[] str;
		return 0;
	}
	cout<<"Error: failed to open file\n:"<<filename<<endl;
	return -1;
}

int matrixbuild(){
	srand(time(NULL));
	int i, j;
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++){
			matrix_0[i][j]=(cl_float)(((cl_float)rand() / RAND_MAX)-0.5)*100;
			matrix_1[i][j]=(cl_float)(((cl_float)rand() / RAND_MAX)-0.5)*100;
			//matrix_0[i][j]=matrix_1[i][j]=i*10+j;
		}
	return MSUCCEES;
}

int runCPU(){
	int i, j;
	int a;
	float b;
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++){
			b=0;
			for(a=0;a<NX;a++)b+=matrix_0[i][a]*matrix_1[a][j];
			matrix_2[i][j]=b;
		}
	return MSUCCEES;
}

int runGPU(){
	size_t size = (NX*NY) * sizeof(cl_float);

	cl_uint numPlatforms;//the NO. of platforms
	cl_platform_id platform = NULL;//the chosen platform
	cl_int	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if (status != CL_SUCCESS)
	{
		cout<<"Error: Getting platforms!"<<endl;
		return 1;
	}

	if(numPlatforms > 0)
	{
		cl_platform_id* platforms = (cl_platform_id* )malloc(numPlatforms* sizeof(cl_platform_id));
		status = clGetPlatformIDs(numPlatforms, platforms, NULL);
		platform = platforms[0];
		free(platforms);
	}

	cl_uint				numDevices = 0;
	cl_device_id        *devices;
	status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);

	if (numDevices == 0)
	{
		cout << "No GPU device available."<<endl;
		cout << "Choose CPU as default device."<<endl;
		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &numDevices);	
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, numDevices, devices, NULL);
	}
	else
	{
		devices = (cl_device_id*)malloc(numDevices * sizeof(cl_device_id));

		status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, numDevices, devices, NULL);

	}
	
	cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, &status);

	cl_command_queue commandQueue = clCreateCommandQueue(context, devices[0], 0, &status);
	
	cl_mem BufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &status);
	cl_mem BufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &status);
	cl_mem BufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY , size, NULL, &status);

	status = clEnqueueWriteBuffer(commandQueue, BufferA, CL_FALSE, 0, size, (cl_float*)matrix_0, 0, NULL, NULL);
	status = clEnqueueWriteBuffer(commandQueue, BufferB, CL_FALSE, 0, size, (cl_float*)matrix_1, 0, NULL, NULL);
	
	
	const char *filename = "MatrixMultiple_kernel.cl";
	string sourceStr;
	status = convertToString(filename, sourceStr);
	const char *source = sourceStr.c_str();
	size_t sourceSize[] = {strlen(source)};
	cl_program program = clCreateProgramWithSource(context, 1, &source, sourceSize, &status);
	status = clBuildProgram(program, numDevices, devices, NULL, NULL, NULL);
	cl_kernel kernel = NULL;
	kernel = clCreateKernel(program, "matrixmultiple", &status);
	
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &BufferA);
	status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &BufferB);
	status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &BufferC);

	size_t globalWorkSize[1];
	globalWorkSize[0] = UNUM;

	T3 = clock();
	status = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, globalWorkSize, NULL, 0, NULL, NULL);

	cl_float *matrix=(cl_float*)malloc(size);
	clEnqueueReadBuffer(commandQueue, BufferC, CL_TRUE, 0, size, matrix, 0, NULL, NULL);
	T4 = clock();

	int i, j;
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++)
			matrix_3[i][j]=matrix[i*NX+j];

	status = clReleaseKernel(kernel);
	status = clReleaseProgram(program);
	status = clReleaseMemObject(BufferA);
	status = clReleaseMemObject(BufferB);
	status = clReleaseMemObject(BufferC);
	status = clReleaseCommandQueue(commandQueue);
	status = clReleaseContext(context);
		
	return MSUCCEES;
}

int printresult(){
	int i, j;
	cout<<"[ ";
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++){
			cout<<matrix_0[i][j]<<" ";
		}
	cout<<"]"<<endl;
	cout<<"[ ";
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++){
			cout<<matrix_1[i][j]<<" ";
		}
	cout<<"]"<<endl;
	cout<<"[ ";
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++){
			cout<<matrix_2[i][j]<<" ";
		}
	cout<<"]"<<endl;
	cout<<"[ ";
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++){
			cout<<matrix_3[i][j]<<" ";
		}
	cout<<"]"<<endl;
	return MSUCCEES;
}

float check(){
	int i, j;
	float b=0;
	for(i=0;i<NX;i++)
		for(j=0;j<NY;j++)
			if(matrix_2[i][j]!=matrix_3[i][j])b++;
	return (1.00-b/NX/NY/1.00)*100;
}

void pause(){
	char N;
	cin>>N; 
}

int main(){
	cout<<"#### Matrix Multiple Program ####"<<endl;
	cout<<"Build the Matrixs(Size:"<<NX<<" x "<<NY<<")"<<endl;
	matrixbuild();
	cout<<"--------------------------------"<<endl;
	cout<<"Multiple with CPU"<<endl;
	T1=clock();
	runCPU();
	T2=clock();
	cout<<"Total Runtime£º " <<(double)(T2-T1) * 1000.0 / CLOCKS_PER_SEC<<" ms!"<< endl;
	cout<<"--------------------------------"<<endl;
	cout<<"Multiple with GPU(Unit Num: "<<UNUM<<")"<<endl;
	T1=clock();
	runGPU();
	T2=clock();
	cout<<"Total Runtime£º " <<(double)(T2-T1) * 1000.0 / CLOCKS_PER_SEC<<" ms!"<< endl;
	cout<<"--------------------------------"<<endl;
	cout<<"Result Check:"<<check()<<" %"<<endl;
	printresult();
	pause();
}