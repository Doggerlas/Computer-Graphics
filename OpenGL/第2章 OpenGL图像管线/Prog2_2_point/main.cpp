#include <GL\glew.h>
#include <GLFW\glfw3.h>
#include <iostream>
using namespace std;

#define numVAOs 1

GLuint renderingProgram;
GLuint vao[numVAOs];

GLuint createShaderProgram() {
	//GLSL顶点着色器代码
	const char *vshaderSource =
		"#version 430    \n"
		"void main(void) \n"
		"{ gl_Position = vec4(0.0, 0.0, 0.0, 1.0); }";
	//GLSL片段着色器代码
	const char *fshaderSource =
		"#version 430    \n"
		"out vec4 color; \n"
		"void main(void) \n"
		"{ color = vec4(0.0, 0.0, 1.0, 1.0); }";

	GLuint vShader = glCreateShader(GL_VERTEX_SHADER); //创建了类型为GL_VERTEX_SHADER的着色器
	GLuint fShader = glCreateShader(GL_FRAGMENT_SHADER);//创建了类型为GL_FRAGMENT_SHADER的着色器
	GLuint vfprogram = glCreateProgram();//创建程序对象

	glShaderSource(vShader, 1, &vshaderSource, NULL);//将GLSL代码从字符串载入空着色器对象中
	glShaderSource(fShader, 1, &fshaderSource, NULL);
	glCompileShader(vShader);//编译各着色器
	glCompileShader(fShader);

	glAttachShader(vfprogram, vShader);//将着色器加入程序对象
	glAttachShader(vfprogram, fShader);
	glLinkProgram(vfprogram);//请求GLSL编译器确保它们的兼容性

	return vfprogram;
}

void init(GLFWwindow* window) {
	renderingProgram = createShaderProgram();
	glGenVertexArrays(numVAOs, vao);//创建OpenGL要求的VAO
	glBindVertexArray(vao[0]);
}

void display(GLFWwindow* window, double currentTime) {
	glUseProgram(renderingProgram);//将含有两个已编译着色器 的程序载入OpenGL管线阶段（在GPU上）
	glPointSize(30.0f);
	glDrawArrays(GL_POINTS, 0, 1);//启动管线处理过程,原始类型是GL_POINTS，仅用来显示一个点
}

int main(void) {
	if (!glfwInit()) { exit(EXIT_FAILURE); }
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(600, 600, "Chapter 2 - program 2", NULL, NULL);
	glfwMakeContextCurrent(window);
	if (glewInit() != GLEW_OK) { exit(EXIT_FAILURE); }
	glfwSwapInterval(1);

	init(window);

	while (!glfwWindowShouldClose(window)) {
		display(window, glfwGetTime());
		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwDestroyWindow(window);
	glfwTerminate();
	exit(EXIT_SUCCESS);
}
