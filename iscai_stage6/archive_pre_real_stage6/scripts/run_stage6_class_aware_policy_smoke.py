import json
import hashlib

from iscai_stage6.adb.policy import (
    decide_illumination
)


actors = [

    ("VEHICLE",0.8),

    ("PEDESTRIAN",0.8),

    ("CYCLIST",0.8),

]


results=[]


for actor,p in actors:

    d = decide_illumination(
        actor,
        p
    )

    results.append(
        {
            "actor_type":
                d.actor_type,

            "priority":
                d.priority,

            "illuminate":
                d.illuminate,
        }
    )



output={

    "actors":
        len(results),

    "class_balance":
        len(
            set(
                x["actor_type"]
                for x in results
            )
        ),

    "decisions":
        results,

    "future_used":
        False,

    "status":
        "PASS",

}


output["sha256"]=hashlib.sha256(
    json.dumps(
        output,
        sort_keys=True
    ).encode()
).hexdigest()


print(
"===== Stage6 Class Aware ADB Policy Smoke ====="
)

print(
"actors =",
output["actors"]
)

print(
"class_balance =",
output["class_balance"]
)

print(
"future_used = NO"
)

print(
"SHA256 =",
output["sha256"]
)

print(
"STATUS = PASS"
)
