//���Ҫ����python���ļ�����Ҫ������->����->���������н����Ϊdll�����ɺ���release�ļ����½��ļ�����Ϊpyd����
#include <iostream>
#include<Eigen/Dense>
#include<Eigen/SVD>
#include<Eigen/sparse>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
namespace py = pybind11;
using namespace std;
using namespace Eigen;


py::array_t<float> get_similar_matrix(py::array_t<float>& origin_point, py::array_t<float>& transformed_point) {

	py::array_t<float> result = py::array_t<float>(4);
	auto r3 = result.mutable_unchecked<1>();
	vector<Eigen::Triplet<float>> a;
	a.reserve(origin_point.shape()[0] * 2);
	auto r = origin_point.unchecked<2>();
	auto r1 = transformed_point.unchecked<2>();
	for (int i = 0; i < origin_point.shape()[0]; i++) {
		a.emplace_back(i*2,0,r(i,0));
		a.emplace_back(i*2, 1, -r(i, 1));
		a.emplace_back(i*2, 2, 1);
		a.emplace_back(i*2 + 1, 0, r(i, 1));
		a.emplace_back(i*2 + 1, 1, r(i, 0));
		a.emplace_back(i*2 + 1, 3, 1);
	}
	VectorXd b = VectorXd::Zero(origin_point.shape()[0] * 2);
	VectorXd x = VectorXd::Zero(4);
	for (int i = 0; i < origin_point.shape()[0]; i++) {
		b[i*2] = (r1(i,0));
		b[i*2+1] = (r1(i, 1));
	}

	LeastSquaresConjugateGradient<SparseMatrix<double>> lscg;
	SparseMatrix<double> A(origin_point.shape()[0] * 2, 4);
	A.setFromTriplets(a.begin(), a.end());
	lscg.compute(A);
	x = lscg.solve(b);
	for (int k = 0; k < 4; k++) {
		r3(k) = x[k];
	}
	return result;
}

py::array_t<double> get_transformed_vertices(py::array_t<float>& vertices, py::array_t<float>& triangle, py::array_t<float>
	& triangle_coefficient, py::array_t<float> neighbors, py::array_t<float>& feature_mesh, py::array_t<float>& feature_mesh_coefficient, py::array_t<float>& dst_feature_point) {
	auto r1 = vertices.unchecked<3>();
	auto r2 = triangle.unchecked<3>();
	auto r3 = triangle_coefficient.unchecked<2>();
	auto r4 = neighbors.unchecked<2>();
	auto r5 = feature_mesh.unchecked<3>();
	auto r6 = feature_mesh_coefficient.unchecked<2>();
	auto r7 = dst_feature_point.unchecked<2>();

	py::array_t<float> result = py::array_t<float>(vertices.shape()[0]*vertices.shape()[1]*2);
	auto r8 = result.mutable_unchecked<1>();

	int width = vertices.shape()[1];
	int height = vertices.shape()[0];
	//ÿһ����������һ��Լ���ÿһ���ı�����һ��Լ���ÿһ���ھӽڵ�Ҳ��һ��Լ����
	vector<Eigen::Triplet<float>> m;
	//ÿһ��������3���㣬ÿ��������x��y����Ҫ����6�Σ��ı�������Ҫ����8�Σ��ھӽڵ�����Ҫ����2��
	int rows = triangle.shape()[0] * 10 + feature_mesh.shape()[0] * 8 + neighbors.shape()[0] * 2;
	m.reserve(rows);
	int dimension = 2;
	int index = 0;
	//���������Լ��
	for (int t = 0; t < triangle.shape()[0]; t++) {
		for (int dim = 0; dim < dimension; dim++) {
			float u = r3(t, 0);
			float v = r3(t, 1);
			int vertice_a = r2(t, 0, 0) * width + r2(t, 0, 1);
			int vertice_b = r2(t, 1, 0) * width + r2(t, 1, 1);
			int vertice_c = r2(t, 2, 0) * width + r2(t, 2, 1);
			if (dim == 0) {
				m.emplace_back(index, vertice_a * 2, 1);
				m.emplace_back(index, vertice_b * 2, u - 1);
				m.emplace_back(index, vertice_c * 2, -u);
				m.emplace_back(index, vertice_c * 2 + 1, -v);
				m.emplace_back(index, vertice_b * 2 + 1, v);
				index++;
			}
			else
			{
				m.emplace_back(index, vertice_a * 2 + 1, 1);
				m.emplace_back(index, vertice_b * 2 + 1, u - 1);
				m.emplace_back(index, vertice_c * 2 + 1, -u);
				m.emplace_back(index, vertice_b * 2, -v);
				m.emplace_back(index, vertice_c * 2, v);
				index++;
			}
		}
		
	}


	//����ı���Լ��
	for (int me = 0; me < feature_mesh.shape()[0]; me++) {
		for (int dim = 0; dim < dimension; dim++) {
			//˳ʱ��a,b,c,d�ĸ�����
			int vertice_a = r5(me, 0, 0) * width + r5(me, 0, 1);
			int vertice_b = r5(me, 1, 0) * width + r5(me, 1, 1);
			int vertice_c = r5(me, 2, 0) * width + r5(me, 2, 1);
			int vertice_d = r5(me, 3, 0) * width + r5(me, 3, 1);
			float u = r6(me, 0);
			float v = r6(me, 1);

			m.emplace_back(index, vertice_a * 2 + dim, 1 - u - v - u * v);
			m.emplace_back(index, vertice_b * 2 + dim, u - u * v);
			m.emplace_back(index, vertice_c * 2 + dim, u * v);
			m.emplace_back(index, vertice_d * 2 + dim, v - u * v);
			index++;
		}
		
	}
	//����ھӶ���Լ��
	for (int n = 0; n < neighbors.shape()[0]; n++) {
		for (int dim = 0; dim < 2; dim++) {
			int vertice = r4(n, 0) * width + r4(n, 1);
			m.emplace_back(index, vertice * 2 + dim, 1);
			index++;
		}
	}
	VectorXd b = VectorXd::Zero(triangle.shape()[0] * 2 + feature_mesh.shape()[0] * 2 + neighbors.shape()[0] * 2);
	VectorXd x = VectorXd::Zero(vertices.shape()[0]*vertices.shape()[1]*2);
	for (int ii = triangle.shape()[0]; ii < triangle.shape()[0] + feature_mesh.shape()[0]; ii++) {
		b[ii * 2] = r7(ii - triangle.shape()[0], 0);
		b[ii * 2 + 1] = r7(ii - triangle.shape()[0], 1);
	}
	for (int jj = triangle.shape()[0] + feature_mesh.shape()[0]; jj < triangle.shape()[0] + feature_mesh.shape()[0] + neighbors.shape()[0]; jj++) {
		
		b[jj* 2] = r1(r4(jj- triangle.shape()[0] + feature_mesh.shape()[0],0),r4(jj - triangle.shape()[0] + feature_mesh.shape()[0],1),0);
		b[jj * 2 + 1] =r1(r4(jj - triangle.shape()[0] + feature_mesh.shape()[0], 0), r4(jj - triangle.shape()[0] + feature_mesh.shape()[0], 1), 1) ;
	}

	LeastSquaresConjugateGradient<SparseMatrix<double>> lscg;
	SparseMatrix<double> A(index, vertices.shape()[0]*vertices.shape()[1]*2);
	A.setFromTriplets(m.begin(), m.end());
	lscg.compute(A);
	x = lscg.solve(b);
	for (int k = 0; k < vertices.shape()[0]*vertices.shape()[1]*2; k++) {
		r8(k) = x[k];
	}
	return result;

}
PYBIND11_MODULE(MSE,m) {
	m.def("get_similar_matrix", &get_similar_matrix);
	m.def("get_transformed_vertices", &get_transformed_vertices);
}