#pragma once

#include <vector>
#include <string>

template<class IteratorType>
void print_arr(IteratorType it, IteratorType end, char end_ch='\n') {
    while(it != end) {
        std::cout << *it << " ";
        it++;
    }
    if(end_ch) {
        std::cout << end_ch;
    }
}