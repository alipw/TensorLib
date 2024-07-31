#include <climits>
#include <iostream>
#include <string>
#include <vector>

#define ti Tensor<int>
#define tl Tensor<int64_t>
#define td Tensor<double>
#define tf Tensor<float>

template <typename T> class Tensor {

public:
  T val;
  std::pair<Tensor<T> *, Tensor<T> *> prev;
  char op;
  double grad; // grad = d{output} /  d{current}

  Tensor(T val = 0, std::pair<Tensor<T> *, Tensor<T> *> prev = {nullptr, nullptr},
         char op = '0', double grad = 1) {
    this->val = val;
    this->prev = prev;
    this->op = op;
    this->grad = grad;
  }

  Tensor operator+(Tensor &other) {
    this->grad = 1;
    other.grad = 1;
    return Tensor(this->val + other.val, {this, &other}, '+');
  }
  Tensor operator-(Tensor &other) {
    this->grad = 1;
    other.grad = 1;
    return Tensor(this->val - other.val, {this, &other}, '-');
  }
  Tensor operator*(Tensor &other) {
    this->grad = other.val;
    other.grad = this->val;
    return Tensor(this->val * other.val, {this, &other}, '*');
  }
  Tensor operator/(Tensor &other) {
    this->grad = 1/other.val;
    other.grad = this->val;
    return Tensor(this->val / other.val, {this, &other}, '/');
  }
};

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &tensor) {
  return os << "Tensor.val: " << tensor.val << std::endl
            << "Tensor.prev: {"
            << (tensor.prev.first == nullptr
                    ? "null"
                    : std::to_string(tensor.prev.first->val))
            << ", "
            << (tensor.prev.second == nullptr
                    ? "null"
                    : std::to_string(tensor.prev.second->val))
            << "}\n"
            << "Tensor.op: " << tensor.op << std::endl
            << "Tensor.grad: " << tensor.grad << std::endl
            << "addr: " << &tensor << std::endl;
}

template <typename T> void print_backward(Tensor<T> *tensor) {
  using namespace std;

  if (tensor == nullptr)
    return;
  cout << *tensor << endl;
  print_backward(tensor->prev.first);
  print_backward(tensor->prev.second);
}

int main() {
  using namespace std;
  td a = td(-3);
  td b = td(-2);
  td d = a / b;
  print_backward(&d);
}
