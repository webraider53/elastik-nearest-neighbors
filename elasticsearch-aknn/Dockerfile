FROM docker.elastic.co/elasticsearch/elasticsearch:6.3.2

ADD . /aknn
WORKDIR /aknn

RUN yum install -y chkconfig
# Install Java 11
# Before building this image, download the .rpm files to elasticsearch-aknn directory
#     from http://www.oracle.com/technetwork/java/javase/downloads/index.html
RUN [ -f jdk-11.0.2_linux-x64_bin.rpm ] && echo "Found" || wget --no-cookies --no-check-certificate --header "Cookie: oraclelicense=accept-securebackup-cookie"   "http://download.oracle.com/otn-pub/java/jdk/11.0.2+9/f51449fcd52f4d52b93a989c5c56ed3c/jdk-11.0.2_linux-x64_bin.rpm"
RUN yum -y install jdk-11.0.2_linux-x64_bin.rpm && \
	alternatives --set java /usr/java/jdk-11.0.2/bin/java
ENV JAVA_HOME=/usr/java/jdk-11.0.2/

# Install gradle 4.9

RUN [ -f gradle gradle-4.9-bin.zip ] && echo "Found" || wget https://services.gradle.org/distributions/gradle-4.9-bin.zip
RUN mkdir /opt/gradle && \
	unzip -d /opt/gradle gradle-4.9-bin.zip
ENV PATH=$PATH:/opt/gradle/gradle-4.9/bin


# Build & install the plugin

RUN gradle clean build -x integTestRunner -x test && \
	elasticsearch-plugin install -b file:build/distributions/elasticsearch-aknn-0.0.1-SNAPSHOT.zip
