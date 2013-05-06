/*
 * debug.cpp
 *
 *  Created on: May 6, 2013
 *      Author: Roman Klapaukh
 */

#include <stdarg.h>
#include <stdio.h>

#include "debug.h"

void debug(const char* text, ...){
#ifdef DEBUG
	va_list a;
	va_start(a, text);
	vprintf(text,a);
	va_end(a);
#endif
}
