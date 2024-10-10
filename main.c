#include "tests.h"

int main() {
    for(int i = 10; i < 100000; i += 10000){
        run_tests(i);
    }
    return 0;
}