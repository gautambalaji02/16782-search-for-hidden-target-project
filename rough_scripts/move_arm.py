#!/usr/bin/env python3
"""
Move the Kinova Gen3 arm by RELATIVE joint angles (degrees).

Usage:
    python3 move_arm.py d1 d2 d3 d4 d5 d6 [--ip 192.168.2.9] [--speed 20]

Example (move joint 4 by +1 deg and joint 6 by +2 deg):
    python3 move_arm.py 0 0 0 1 0 2
"""

import argparse
import sys
import threading

from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.messages import Session_pb2, Base_pb2
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient

DEFAULT_IP = "192.168.2.9"
USER, PASSWORD = "admin", "admin"   # change if you changed the web-app password
TIMEOUT_S = 30                      # max time to wait for the motion to finish


def parse_args():
    p = argparse.ArgumentParser(description="Move Gen3 by relative joint angles (deg).")
    p.add_argument("deltas", nargs="+", type=float,
                   help="relative joint angles in degrees, one per joint (6 for a Gen3 6-DOF)")
    p.add_argument("--ip", default=DEFAULT_IP, help=f"robot IP (default {DEFAULT_IP})")
    p.add_argument("--speed", type=float, default=20.0,
                   help="joint speed limit in deg/s (default 20)")
    p.add_argument("--dry-run", action="store_true",
                   help="print the target angles but do not move the arm")
    return p.parse_args()


def make_action_done_event(base):
    """Subscribe to action notifications; returns (event, handle, result_holder)."""
    done = threading.Event()
    result = {"event": None}

    def callback(notification):
        ev = notification.action_event
        if ev in (Base_pb2.ACTION_END, Base_pb2.ACTION_ABORT):
            result["event"] = ev
            done.set()

    handle = base.OnNotificationActionTopic(callback, Base_pb2.NotificationOptions())
    return done, handle, result


def main():
    args = parse_args()
    deltas = args.deltas

    transport = TCPTransport()
    router = RouterClient(transport, lambda e: print("Router error:", e))
    transport.connect(args.ip, 10000)

    info = Session_pb2.CreateSessionInfo()
    info.username = USER
    info.password = PASSWORD
    info.session_inactivity_timeout = 60000      # ms
    info.connection_inactivity_timeout = 2000    # ms

    session = SessionManager(router)
    session.CreateSession(info)

    try:
        base = BaseClient(router)

        n_joints = base.GetActuatorCount().count
        if len(deltas) != n_joints:
            print(f"Error: this arm has {n_joints} joints, but {len(deltas)} "
                  f"angle(s) were given.", file=sys.stderr)
            return 2

        # Make sure we are in single-level (high-level) servoing mode.
        servo = Base_pb2.ServoingModeInformation()
        servo.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        base.SetServoingMode(servo)

        # Read current angles (Kortex reports them in [0, 360) deg).
        current = [j.value for j in base.GetMeasuredJointAngles().joint_angles]
        target = [(c + d) % 360.0 for c, d in zip(current, deltas)]

        print("joint :   current  ->   target   (delta)")
        for i, (c, t, d) in enumerate(zip(current, target, deltas)):
            print(f"  {i}   : {c:8.2f}  -> {t:8.2f}  ({d:+.2f})")

        if args.dry_run:
            print("\n--dry-run: no motion commanded.")
            return 0

        # Build the reach-joint-angles action.
        action = Base_pb2.Action()
        action.name = "relative joint move"
        action.application_data = ""
        for i, t in enumerate(target):
            ja = action.reach_joint_angles.joint_angles.joint_angles.add()
            ja.joint_identifier = i
            ja.value = t
        action.reach_joint_angles.constraint.type = Base_pb2.JOINT_CONSTRAINT_SPEED
        action.reach_joint_angles.constraint.value = args.speed

        done, notif_handle, result = make_action_done_event(base)
        try:
            print("\nMoving...")
            base.ExecuteAction(action)
            if not done.wait(TIMEOUT_S):
                print(f"Timed out after {TIMEOUT_S}s waiting for the motion to finish.",
                      file=sys.stderr)
                base.StopAction()
                return 1
            if result["event"] == Base_pb2.ACTION_ABORT:
                print("Motion was aborted by the robot.", file=sys.stderr)
                return 1
            print("Motion finished.")
        finally:
            base.Unsubscribe(notif_handle)

        final = [j.value for j in base.GetMeasuredJointAngles().joint_angles]
        print("Final joint angles (deg):")
        for i, v in enumerate(final):
            print(f"  joint {i}: {v:.2f}")
        return 0

    finally:
        session.CloseSession()
        transport.disconnect()


if __name__ == "__main__":
    sys.exit(main())
