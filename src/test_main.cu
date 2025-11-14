/*
 * Simple test program for W4A16 SM90 kernel
 */

#include <stdio.h>

// Declare the test function
extern "C" void test_w4a16_sm90_kernel();

int main() {
    printf("Testing W4A16 Hopper (SM90) Kernel...\n");
    printf("=====================================\n\n");

    test_w4a16_sm90_kernel();

    printf("\nKernel library compiled and linked successfully!\n");
    printf("Library location: lib/libw4a16_sm90_kernel.so\n");

    return 0;
}
