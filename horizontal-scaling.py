
import boto3
import botocore
import os
import requests
import time
import json
import configparser
import re
from dateutil.parser import parse
from datetime import datetime


########################################
# Constants
########################################
with open('horizontal-scaling-config.json') as file:
    configuration = json.load(file)

LOAD_GENERATOR_AMI = configuration['load_generator_ami']
WEB_SERVICE_AMI = configuration['web_service_ami']
INSTANCE_TYPE = configuration['instance_type']
sg1_Description='Permissions for Load Generator'
sg1_GroupName='LoadGenerator HTTP Sec Group'
sg2_Description='Permissions for Web Service'
sg2_GroupName='Web Service HTTP Sec Group'

# Credentials fetched from environment variables
SUBMISSION_USERNAME = os.environ['SUBMISSION_USERNAME']
SUBMISSION_PASSWORD = os.environ['SUBMISSION_PASSWORD']

########################################
# Tags
########################################
tag_pairs = [
    ("Project", "vm-scaling"),
]
TAGS = [{'Key': k, 'Value': v} for k, v in tag_pairs]

TEST_NAME_REGEX = r'name=(.*log)'

########################################
# Utility functions
########################################


def create_instance(ami, sg_name):
    """
    Given AMI, create and return an AWS EC2 instance object
    :param ami: AMI image name to launch the instance with
    :param sg_name: name of the security group to be attached to instance
    :return: instance object
    """
    ec2 = boto3.resource('ec2',region_name="us-east-1")
    instance = ec2.create_instances(
        ImageId=ami,
        InstanceType=INSTANCE_TYPE,
        MaxCount=1,
        MinCount=1,
        SecurityGroups=[
        sg_name,
        ],
        TagSpecifications=[{
        'ResourceType': 'instance',
        'Tags': TAGS
    }, {
        'ResourceType': 'volume',
        'Tags': TAGS
    }])[0]
    instance.wait_until_running()
    instance.reload()

    # TODO: Create an EC2 instance
    # Wait for the instance to enter the running state
    # Reload the instance attributes

    return instance


def initialize_test(lg_dns, first_web_service_dns):
    """
    Start the horizontal scaling test
    :param lg_dns: Load Generator DNS
    :param first_web_service_dns: Web service DNS
    :return: Log file name
    """

    add_ws_string = 'http://{}/test/horizontal?dns={}'.format(
        lg_dns, first_web_service_dns
    )
    response = None
    while not response or response.status_code != 200:
        try:
            response = requests.get(add_ws_string)
        except requests.exceptions.ConnectionError:
            time.sleep(1)
            pass 

    # TODO: return log File name
    return get_test_id(response)

def print_section(msg):
    """
    Print a section separator including given message
    :param msg: message
    :return: None
    """
    print(('#' * 40) + '\n# ' + msg + '\n' + ('#' * 40))


def get_test_id(response):
    """
    Extracts the test id from the server response.
    :param response: the server response.
    :return: the test name (log file name).
    """
    response_text = response.text

    regexpr = re.compile(TEST_NAME_REGEX)

    return regexpr.findall(response_text)[0]


def is_test_complete(lg_dns, log_name):
    """
    Check if the horizontal scaling test has finished
    :param lg_dns: load generator DNS
    :param log_name: name of the log file
    :return: True if Horizontal Scaling test is complete and False otherwise.
    """

    log_string = 'http://{}/log?name={}'.format(lg_dns, log_name)

    # creates a log file for submission and monitoring
    f = open(log_name + ".log", "w")
    log_text = requests.get(log_string).text
    f.write(log_text)
    f.close()

    return '[Test finished]' in log_text


def add_web_service_instance(lg_dns, sg2_name, log_name):
    """
    Launch a new WS (Web Server) instance and add to the test
    :param lg_dns: load generator DNS
    :param sg2_name: name of WS security group
    :param log_name: name of the log file
    """
    ins = create_instance(WEB_SERVICE_AMI, sg2_name)
    print("New WS launched. id={}, dns={}".format(
        ins.instance_id,
        ins.public_dns_name)
    )
    add_req = 'http://{}/test/horizontal/add?dns={}'.format(
        lg_dns,
        ins.public_dns_name
    )
    while True:
        if requests.get(add_req).status_code == 200:
            print("New WS submitted to LG.")
            break
        elif is_test_complete(lg_dns, log_name):
            print("New WS not submitted because test already completed.")
            break


def authenticate(lg_dns, submission_password, submission_username):
    """
    Authentication on LG
    :param lg_dns: LG DNS
    :param submission_password: SUBMISSION_PASSWORD
    :param submission_username: SUBMISSION_USERNAME
    :return: None
    """

    authenticate_string = 'http://{}/password?passwd={}&username={}'.format(
        lg_dns, submission_password, submission_username
    )
    response = None
    while not response or response.status_code != 200:
        try:
            response = requests.get(authenticate_string)
            break
        except requests.exceptions.ConnectionError:
            pass


def get_rps(lg_dns, log_name):
    """
    Return the current RPS as a floating point number
    :param lg_dns: LG DNS
    :param log_name: name of log file
    :return: latest RPS value
    """

    log_string = 'http://{}/log?name={}'.format(lg_dns, log_name)
    config = configparser.ConfigParser(strict=False)
    config.read_string(requests.get(log_string).text)
    sections = config.sections()
    sections.reverse()
    rps = 0
    for sec in sections:
        if 'Current rps=' in sec:
            rps = float(sec[len('Current rps='):])
            break
    return rps


def get_test_start_time(lg_dns, log_name):
    """
    Return the test start time in UTC
    :param lg_dns: LG DNS
    :param log_name: name of log file
    :return: datetime object of the start time in UTC
    """
    log_string = 'http://{}/log?name={}'.format(lg_dns, log_name)
    start_time = None
    while start_time is None:
        config = configparser.ConfigParser(strict=False)
        config.read_string(requests.get(log_string).text)
        # By default, options names in a section are converted
        # to lower case by configparser
        start_time = dict(config.items('Test')).get('starttime', None)
    return parse(start_time)

def security_group_id(sg_des,sg_name,sg_permissions):
    # make a new security with given ddescription and group name
    #return the security group id
    ec2 = boto3.client('ec2',region_name="us-east-1")
    sg_group= ec2.create_security_group(
        Description=sg_des,
        GroupName=sg_name,
        TagSpecifications=[{
                'ResourceType': 'security-group',
                'Tags': TAGS
        }])
    ec2.authorize_security_group_ingress(
        GroupId=sg_group['GroupId'],
        IpPermissions=sg_permissions)
    return sg_group['GroupId']


########################################
# Main routine
########################################
def main():
    # BIG PICTURE TODO: Provision resources to achieve horizontal scalability
    #   - Create security groups for Load Generator and Web Service
    #   - Provision a Load Generator instance
    #   - Provision a Web Service instance
    #   - Register Web Service DNS with Load Generator
    #   - Add Web Service instances to Load Generator
    #   - Terminate resources

    print_section('1 - create two security groups')
    sg_permissions = [
        {'IpProtocol': 'tcp',
         'FromPort': 80,
         'ToPort': 80,
         'IpRanges': [{'CidrIp': '0.0.0.0/0'}],
         'Ipv6Ranges': [{'CidrIpv6': '::/0'}],
         }
    ]

    # TODO: Create two separate security groups and obtain the group ids
    sg1_id = security_group_id(sg1_Description,sg1_GroupName,sg_permissions)
      # Security group for Load Generator instances
    sg2_id  = security_group_id(sg2_Description,sg2_GroupName,sg_permissions)# Security group for Web Service instances

    print_section('2 - create LG')

    # TODO: Create Load Generator instance and obtain ID and DNS
    lg = create_instance(LOAD_GENERATOR_AMI, sg1_GroupName)
    lg_id = lg.instance_id
    lg_dns = lg.public_dns_name
    print("Load Generator running: id={} dns={}".format(lg_id, lg_dns))

    print_section('3. Authenticate with the load generator')
    authenticate(lg_dns, SUBMISSION_PASSWORD, SUBMISSION_USERNAME)

    # TODO: Create First Web Service Instance and obtain the DNS
    web_service_dns = create_instance(WEB_SERVICE_AMI, sg2_GroupName).public_dns_name

    print_section('4. Submit the first WS instance DNS to LG, starting test.')
    log_name = initialize_test(lg_dns, web_service_dns)
    #last_launch_time = get_test_start_time(lg_dns, log_name)
    last_launch_time = datetime.now()
    while not is_test_complete(lg_dns, log_name):
        # TODO: Check RPS and last launch time
        rps = get_rps(lg_dns, log_name)
        current_time = datetime.now()
        # TODO: Add New Web Service Instance if Required
        if ((current_time - last_launch_time).total_seconds() > 100 ):
            add_web_service_instance(lg_dns, sg2_GroupName, log_name)
            last_launch_time = current_time
        time.sleep(1)

    print_section('End Test')

    # TODO: Terminate Resources
    #boto3.client.terminate_instances(InstanceIds=lg.instance_id)


if __name__ == '__main__':
    main()
