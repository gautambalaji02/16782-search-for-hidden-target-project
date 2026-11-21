import sys
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

IP = sys.argv[1] if len(sys.argv) > 1 else "192.168.2.9"
USER, PASSWORD = "admin", "admin"   # change if you changed the web-app password

transport = TCPTransport()
router = RouterClient(transport, lambda e: print("Router error:", e))
transport.connect(IP, 10000)

info = Session_pb2.CreateSessionInfo()
info.username = USER
info.password = PASSWORD
info.session_inactivity_timeout = 60000      # ms
info.connection_inactivity_timeout = 2000    # ms

session = SessionManager(router)
session.CreateSession(info)
try:
    base = BaseClient(router)
    print("Connected")
    print("Joint angles (deg):")
    for j in base.GetMeasuredJointAngles().joint_angles:
        print(f"  joint {j.joint_identifier}: {j.value:.2f}")
    pose = base.GetMeasuredCartesianPose()
    print(f"Tool pose: x={pose.x:.3f} y={pose.y:.3f} z={pose.z:.3f} m")
finally:
    session.CloseSession()
    transport.disconnect()
