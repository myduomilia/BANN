CC=g++
CFLAGS = -c -Wall -std=c++14 -fPIC -I /usr/include/eigen3/Eigen
LDFLAGS = -lpthread -lboost_system

SOURCES = sources/main.cpp\
	  sources/core/core.cpp
   
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = bann

	
$(EXECUTABLE): $(OBJECTS) copy_conf
	$(CC) -o build/$(EXECUTABLE) $(OBJECTS) $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@
	
copy_conf:
	cp conf/settings.json build/settings.json

clean:
	rm -rf *.o $(OBJECTS)
